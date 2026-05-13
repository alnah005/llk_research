# 3.3 Attention Masking and Positions

## Context

The dual-expert transformer needs a carefully designed attention mask to enforce
the correct information flow between prefix tokens (images + language) and suffix
tokens (actions). The mask implements prefix-LM semantics: prefix tokens attend
bidirectionally among themselves, while action tokens can see the entire prefix
but the prefix cannot see action tokens. This section traces exactly how the
`mask_ar` array and the cumulative-sum trick produce the 2D attention mask, how
positions are computed, and how the mask changes during cached denoising.

---

## The `mask_ar` Array

The attention pattern is controlled by a 1D boolean/integer array called
`mask_ar` (short for "mask autoregressive"). Each entry indicates whether the
corresponding token introduces a causal break:

- `0` (False): this token shares bidirectional attention with its neighbors
- `1` (True): this token starts a new causal group; previous tokens cannot see it

**File:** `/tmp/openpi/src/openpi/models/pi0.py`, lines 19-44 (JAX version)

```python
def make_attn_mask(input_mask, mask_ar):
    """
    mask_ar examples:
      [[1 1 1 1 1 1]]: pure causal attention.
      [[0 0 0 1 1 1]]: prefix-lm attention (first 3 bidirectional, last 3 causal).
      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

---

## The Cumsum Trick

The `mask_ar` values are converted to a 2D attention mask through a cumulative
sum. Here is how it works step by step:

**Example: Prefix-LM with 4 prefix tokens and 3 action tokens**

```
mask_ar = [0, 0, 0, 0, 1, 0, 0]
            <-- prefix -->  <-- actions -->
```

Step 1: Compute cumulative sum along each position:

```
cumsum  = [0, 0, 0, 0, 1, 1, 1]
```

Step 2: Build the 2D mask via `cumsum[:, None, :] <= cumsum[:, :, None]`:

```
           key positions (S) -->
           0  0  0  0  1  1  1
     q   +-----------------------+
     u  0| T  T  T  T  T  T  T  |   row: cumsum=0, col: cumsum>=0 all True
     e  0| T  T  T  T  T  T  T  |   (prefix sees everything including actions??)
     r  0| T  T  T  T  T  T  T  |
     y  0| T  T  T  T  T  T  T  |
        1| F  F  F  F  T  T  T  |   row: cumsum=1, col: only cumsum>=1 are True
     (T)1| F  F  F  F  T  T  T  |
        1| F  F  F  F  T  T  T  |
          +-----------------------+
```

Wait -- this looks wrong. The prefix (rows 0-3) appears to attend to the action
tokens (columns 4-6). But notice the inequality direction: the test is
`cumsum[key] <= cumsum[query]`. Let me re-derive:

```
attn_mask[query, key] = cumsum[key] <= cumsum[query]
```

For query=0 (prefix, cumsum=0), key=4 (action, cumsum=1):
  `1 <= 0` = False. Prefix CANNOT see action tokens.

For query=4 (action, cumsum=1), key=0 (prefix, cumsum=0):
  `0 <= 1` = True. Action tokens CAN see prefix tokens.

Corrected 2D mask:

```
           key positions (S) -->
           c=0 c=0 c=0 c=0 c=1 c=1 c=1
     q   +-------------------------------+
    c=0  |  T   T   T   T   F   F   F   |  prefix: sees prefix only
    c=0  |  T   T   T   T   F   F   F   |  (bidirectional among prefix)
    c=0  |  T   T   T   T   F   F   F   |
    c=0  |  T   T   T   T   F   F   F   |
    c=1  |  T   T   T   T   T   T   T   |  action: sees everything
    c=1  |  T   T   T   T   T   T   T   |  (bidirectional among actions)
    c=1  |  T   T   T   T   T   T   T   |
         +-------------------------------+
```

This is exactly prefix-LM: prefix tokens attend bidirectionally among
themselves but cannot see action tokens. Action tokens see the entire sequence
bidirectionally.

---

## How `mask_ar` Is Constructed in Pi0.5

The prefix and suffix each contribute their portion of `mask_ar`:

**Prefix construction** (`embed_prefix` in pi0.py, lines 109-136):

```python
# image tokens attend to each other
ar_mask += [False] * image_tokens.shape[1]    # e.g., 256 image tokens

# full attention between image and language inputs
ar_mask += [False] * tokenized_inputs.shape[1]  # e.g., 48 language tokens
```

All prefix tokens get `False` (0), meaning they form a single bidirectional
attention block.

**Suffix construction** (`embed_suffix` in pi0.py, lines 139-186):

```python
# For pi0 (not pi0.5): state token starts a new causal group
ar_mask += [True]  # state token: causal break

# First action token: causal break (prevents prefix from seeing actions)
ar_mask += [True] + ([False] * (self.action_horizon - 1))
```

The critical detail: the first action/state token gets `True`, creating a
causal break. All subsequent action tokens get `False`, so they attend
bidirectionally among themselves. The full `mask_ar` looks like:

```
Prefix (images + language):  [0, 0, 0, ..., 0]   (all bidirectional)
Suffix (state + actions):    [1, 1, 0, 0, ..., 0] (break at state, break at
                                                    first action, rest bidir)
For pi0.5 (no state token):
Suffix (actions only):       [1, 0, 0, ..., 0]    (break at first action,
                                                    rest bidirectional)
```

---

## Resulting Attention Pattern (ASCII)

For a concrete example with 6 prefix tokens and 4 action tokens (pi0.5 mode):

```
mask_ar = [0  0  0  0  0  0 | 1  0  0  0]
cumsum  = [0  0  0  0  0  0 | 1  1  1  1]

              key (S) --->
              img img img lang lang lang | act act act act
       query  c=0 c=0 c=0 c=0  c=0  c=0 | c=1 c=1 c=1 c=1
       (T)  +----------------------------+------------------+
  img  c=0  |  .   .   .   .    .    .   |                  |
  img  c=0  |  .   .   .   .    .    .   |    (blocked)     |
  img  c=0  |  .   .   .   .    .    .   |                  |
  lang c=0  |  .   .   .   .    .    .   |                  |
  lang c=0  |  .   .   .   .    .    .   |                  |
  lang c=0  |  .   .   .   .    .    .   |                  |
       -----+----------------------------+------------------+
  act  c=1  |  .   .   .   .    .    .   |  .   .   .   .   |
  act  c=1  |  .   .   .   .    .    .   |  .   .   .   .   |
  act  c=1  |  .   .   .   .    .    .   |  .   .   .   .   |
  act  c=1  |  .   .   .   .    .    .   |  .   .   .   .   |
            +----------------------------+------------------+

  Legend:  .  = attention allowed      (blank) = attention blocked
```

The pattern divides into four quadrants:
1. **Top-left** (prefix-to-prefix): Fully bidirectional
2. **Top-right** (prefix-to-action): Blocked -- prefix cannot see actions
3. **Bottom-left** (action-to-prefix): Allowed -- actions see all prefix tokens
4. **Bottom-right** (action-to-action): Fully bidirectional among action tokens

---

## Position Computation

Token positions are computed from the `input_mask` (which is `True` for real
tokens and `False` for padding):

**File:** `/tmp/openpi/src/openpi/models/pi0.py`, line 208

```python
positions = jnp.cumsum(input_mask, axis=1) - 1
```

**File:** `/tmp/openpi/src/openpi/models_pytorch/pi0_pytorch.py`, line 344

```python
position_ids = torch.cumsum(pad_masks, dim=1) - 1
```

For a batch element with 6 real prefix tokens and 4 real action tokens (no
padding):

```
input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cumsum     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
positions  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

With padding in the prefix (e.g., shorter language input):

```
input_mask = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
cumsum     = [1, 2, 3, 4, 4, 4, 5, 6, 7, 8]
positions  = [0, 1, 2, 3, 3, 3, 4, 5, 6, 7]
```

Padding tokens get repeated positions (they will be masked out by the
`valid_mask` anyway). This scheme ensures that valid tokens receive consecutive
integer positions starting from 0, which is what RoPE expects.

---

## Denoising KV Cache Mask

During iterative denoising, the mask has a different structure. The prefix K/V
is cached and the suffix generates fresh Q/K/V. The attention mask must let
suffix queries attend to both the cached prefix keys and their own suffix keys:

**File:** `/tmp/openpi/src/openpi/models/pi0.py`, lines 246-257

```python
# suffix_attn_mask: [B, suffix_len, suffix_len] -- suffix self-attention
suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

# prefix_attn_mask: [B, suffix_len, prefix_len] -- suffix attending to prefix
prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])

# full mask: [B, suffix_len, prefix_len + suffix_len]
full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
```

The denoising mask structure:

```
              key (S) --->
              cached prefix K/V     |  fresh suffix K/V
              p0  p1  p2 ... pP     |  s0  s1  s2  s3
       query +----------------------+-------------------+
       (T)   |                      |                   |
  s0         |  .   .   .       .   |  .   .   .   .    |
  s1         |  .   .   .       .   |  .   .   .   .    |
  s2         |  .   .   .       .   |  .   .   .   .    |
  s3         |  .   .   .       .   |  .   .   .   .    |
             +----------------------+-------------------+

  Shape: [B, suffix_len, prefix_len + suffix_len]
```

All suffix tokens can attend to all valid prefix tokens (determined by
`prefix_mask`) and to each other (determined by the suffix's own `mask_ar`
pattern).

**Positions during denoising:**

**File:** `/tmp/openpi/src/openpi/models/pi0.py`, line 259

```python
positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
```

This offsets the suffix positions by the number of valid prefix tokens, so the
suffix positions continue where the prefix left off. For example, if the prefix
has 304 valid tokens and the suffix has 4 action tokens:

```
suffix positions = [304, 305, 306, 307]
```

The PyTorch version computes the same thing:

**File:** `/tmp/openpi/src/openpi/models_pytorch/pi0_pytorch.py`, lines 443-444

```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
```

---

## Padding and Valid Mask

The `make_attn_mask` function also incorporates a validity mask:

```python
valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
return jnp.logical_and(attn_mask, valid_mask)
```

This ensures that padding tokens (where `input_mask=False`) can neither attend
nor be attended to, regardless of the `mask_ar` pattern. The `valid_mask` is a
2D outer product of the 1D `input_mask` with itself.

---

## PyTorch Mask Format

The PyTorch implementation converts the boolean 2D mask to a 4D float mask
compatible with HuggingFace Transformers' additive masking convention:

**File:** `/tmp/openpi/src/openpi/models_pytorch/pi0_pytorch.py`, lines 157-160

```python
def _prepare_attention_masks_4d(self, att_2d_masks):
    att_2d_masks_4d = att_2d_masks[:, None, :, :]  # [B, 1, T, S]
    return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
```

The same `big_neg = -2.3819763e38` sentinel value is used in both JAX and
PyTorch implementations (matching the original Gemma source). The `[:, None, :, :]`
adds the head dimension, which broadcasts across all attention heads.

---

## Key Takeaways

- The `mask_ar` array uses a cumulative-sum trick to convert a 1D annotation
  into a 2D attention mask. Tokens with the same cumsum value attend
  bidirectionally; a higher cumsum value can attend to lower ones but not
  vice versa.

- Prefix tokens all get `mask_ar=0` (cumsum=0): fully bidirectional. The first
  action token gets `mask_ar=1` (cumsum=1): creates a one-way barrier. Remaining
  action tokens get `mask_ar=0` (stay at cumsum=1): bidirectional among
  themselves and with the prefix.

- Positions are computed as `cumsum(input_mask) - 1`, giving consecutive integers
  starting from 0 for valid tokens, with padding tokens receiving repeated
  positions.

- During denoising, the mask shape changes to `[B, suffix_len, prefix_len + suffix_len]`
  since only suffix tokens generate queries. Suffix positions are offset by the
  prefix length to maintain consistent RoPE encoding.

- Both JAX and PyTorch use the same sentinel value (`-2.3819763e38`) for masked
  positions, avoiding infinity while staying in the representable bfloat16 range.
