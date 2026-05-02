# Guide Generation Prompt (Deep Planning Mode)

This file instructs you to generate a structured, multi-chapter markdown guide for a given topic using the deep-planning approach. Multiple independent architects compete on both the structural plan and each chapter's initial content before quality-control agents (B and C) ever see the result.

---

## Shared Research Topics — Trigger Behavior

This flow is **one-shot**. It is triggered manually; it does not poll or loop on its own.

### On startup

```bash
# First time: clone
git clone https://github.com/alnah005/llk_research.git llk_research
# Every time: pull latest
cd llk_research && git pull
```

Read `llk_research/research_topics.md` and collect all topics with `Status: Pending`.

- If there are **no Pending topics**: exit immediately. Nothing to do.
- If there are **Pending topics**: you (the orchestrator) directly manage all topics. There are no coordinator sub-agents. You spawn all agents yourself, directly, for each topic.

### Orchestrator state

For each Pending topic, initialize and track the following state yourself:

```
topic_state = {
    topic_name,
    output_dir,           # <RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/
    tmp_dir,              # /tmp/research_<snake_case_topic_name>/
    skip,                 # true if output_dir already exists
    N,                    # complexity score — set during Step 0, reused for all chapters
    current_chapter,      # index of chapter currently being processed (0-based)
    phase,                # "assess" | "plan" | "write" | "review" | "compress" | "finalpass" | "done"
    pending_b_feedback,   # feedback from last Agent B invocation, or null
    pending_c_feedback,   # CRUCIAL suggestions from last Agent C invocation, or null
}
```

If `output_dir` already exists for a topic, set `skip = true` and advance it directly to the git-push step.

### Dispatch loop

The dispatch loop is **event-driven per topic**, not wave-synchronized across topics. Topics are fully independent — one topic completing a phase never waits on another topic.

At each step:

1. **As soon as any agent completes**, read its result immediately and advance that topic's state.
2. Do not wait for other topics' agents to finish before advancing a topic that is ready.
3. At any moment, up to one agent *sequence* per topic may be running in parallel across topics. Within a topic, deep-planning rounds (Steps 1–4) run as a batch before advancing.
4. **Report the state table** whenever you spawn a new agent (see Reporting format below).
5. The loop ends when all topics are `done`.

### Reporting format

Before each dispatch wave, print the state table in this format:

```
**State:**
<topic_name>:  phase=<phase>, chapter=<N>, architects=<N>  → <Agent X> running
<topic_name>:  phase=<phase>, chapter=<N>, architects=<N>  → <Agent X> running
...
```

This makes the orchestrator's progress visible at every step. Topics that are further ahead (e.g., already in review) should show a different phase/agent than topics still writing.

### Per-topic context passing

You pass each agent only what it needs — no shared context between agents:
- Plan Generators receive: topic scope (Why Needed + Questions), any failure notes from a previous verification round.
- Plan Evaluators receive: all N candidate plans (from `/tmp`).
- Plan Synthesizer receives: all N candidate plans + all N evaluations (from `/tmp`).
- Plan Verifier receives: the synthesized candidate plan (from `/tmp`), any prior failure notes.
- Chapter Generators receive: the verified `plan.md`, the chapter spec, content of all previously completed chapters, any failure notes from a previous verification round.
- Chapter Evaluators receive: all N candidate chapter directories (from `/tmp`), the chapter spec from `plan.md`.
- Chapter Synthesizer receives: all N candidate chapters + all N evaluations (from `/tmp`).
- Chapter Verifier receives: the synthesized candidate chapter (from `/tmp`), the chapter spec from `plan.md`, any prior failure notes.
- Agent B receives: `plan.md` and the file paths of the chapter to review.
- Agent C receives: the chapter file paths and the current pass number.
- Never let B or C see any agent's reasoning. Never let any generator pre-read evaluator or verifier prompts.

### Completion per topic

When a topic's guide satisfies the completion condition, you directly:

1. Update `research_topics.md` for that topic:
   - Set `**Status:**` to `Completed`
   - Add a `**Guide:**` field pointing to the output directory

   Example:
   ```
   **Status:** Completed
   **Guide:** <RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/
   ```

2. Push immediately — do not wait for other topics:

```bash
cd llk_research
git add research_topics.md
git commit -m "research: completed topic <topic-name>"
git push
```

If push fails due to a concurrent push, rebase and retry:

```bash
git pull --rebase && git push
```

### Parallelism rules

- **Across topics**: one topic's agent sequence runs in parallel with another topic's. ✓
- **Within a topic, across chapters**: chapters are processed in order (Chapter 1 first). Do NOT parallelize chapters within a topic.
- **Within a deep-planning round**: all N generators run in parallel, then all N evaluators run in parallel, then Synthesizer, then Verifier — each wave completes before the next starts.
- **Within a chapter, across B/C/A agents**: A → B → A(fix) → C → A(fix) is strictly sequential within each topic after the deep-planning round completes.
- **Agent A, B, and C are always spawned by you (the orchestrator) directly.** No coordinator sub-agents exist.

### Completion

The dispatch loop ends when all topics are `done`. Report back:
- How many topics were researched.
- Topic name, output directory, N chosen, and one-line finding summary for each.

---

## Inputs

- **Topic file**: The topic's `**Why Needed:**` and `**Questions:**` fields from `research_topics.md`, treated as the guide scope and key concepts.
- **Output directory**: `<RESEARCH_OUTPUT_BASE_DIR>/<snake_case_topic_name>/`
- **Temp directory**: `/tmp/research_<snake_case_topic_name>/` — all intermediate artifacts go here

Read both fields before doing anything else.

---

## Step 0 — Difficulty Assessment (Sets N for the Entire Topic)

Before any planning or writing begins, the orchestrator assesses the topic and assigns **N** — the number of parallel architects used for both Phase 0 and every chapter's initial write.

| Difficulty | Criteria | N |
|------------|----------|---|
| **Trivial** | Narrow topic, few questions, well-documented area | 2 |
| **Low** | Focused topic, clear scope, moderate depth | 3 |
| **Medium** | Multi-concept topic, unclear trade-offs, or cross-system interactions | 4 |
| **High** | Broad topic, deep technical complexity, or significant ambiguity | 5 |
| **Critical** | Foundational or architecture-level topic with high coverage risk | 6+ |

The orchestrator MUST report the chosen N and its justification before spawning any agents. N is stored in `topic_state.N` and reused unchanged for every chapter. It is never re-assessed mid-topic.

---

## Phase 0 — Deep Plan Generation

Instead of a single Agent A producing the plan, N independent Plan Generators compete. All intermediate artifacts are written to `/tmp/research_<topic>/` — never to the project directory.

### Step 1: N Plan Generators (Parallel)

Spawn N Plan Generator agents **simultaneously**. Each receives the topic scope (Why Needed + Questions) and any failure notes from a previous verification round, but MUST produce an **independent** plan without being aware of the others.

#### ⚠️ File Location Rule
Plan Generators MUST write to `/tmp`, NOT the output directory:
```
/tmp/research_<topic>/plan_v<1..N>.md
```

Each plan MUST include:
1. **Audience** — who is this guide for, what do they already know.
2. **Chapter list** — 1–8 chapters, ordered foundational to advanced. For each chapter:
   - Chapter number and title
   - One-sentence description
   - List of files inside the chapter directory
   - Bullet points describing each file's content
3. **Conventions** — terminology, notation, formatting rules.
4. **Cross-chapter dependencies** — which chapters reference concepts from earlier ones.

### Step 2: N Plan Evaluators (Parallel)

Spawn N Plan Evaluator agents **simultaneously**, one per candidate plan. Each receives all N candidate plans (read from `/tmp`) and their assigned plan index.

#### ⚠️ File Location Rule
Plan Evaluators MUST write to `/tmp`, NOT the output directory:
```
/tmp/research_<topic>/eval_plan_v<1..N>.md
```

Each evaluation MUST include:
- **Assigned plan summary** (one paragraph)
- **Strengths**: coverage, ordering, audience fit
- **Weaknesses / risks**: gaps, redundancy, poor ordering
- **Comparison to other plans**: what other plans do better or worse
- **Recommendation**: Accept / Accept with modifications / Reject (with reason)

### Step 3: 1 Plan Synthesizer

Spawn **one** Plan Synthesizer with all N plans and all N evaluations (read from `/tmp`).

#### ⚠️ File Location Rule
The Plan Synthesizer MUST write the candidate final plan to `/tmp`, NOT the output directory:
```
/tmp/research_<topic>/plan_final.md
```

The Plan Synthesizer MUST:
1. Review all plans and evaluations
2. Select the single best plan or construct a hybrid
3. Include a **Selection Rationale** section

### Step 4: 1 Plan Verifier

Spawn **one** Plan Verifier with the candidate final plan (read from `/tmp`) and any prior failure notes.

The Plan Verifier MUST:
1. Check that the chapter list is complete, well-ordered, and non-overlapping
2. Check that every `**Questions:**` field item from the topic is covered by at least one chapter
3. Check for cross-chapter dependency cycles or unreachable content
4. Return a verdict:

**VERDICT: PASS** — the plan is solid. Provide a brief rationale.

**VERDICT: FAIL** — the plan has unresolved issues. List each issue precisely:
```
ISSUE 1: [Description of gap or flaw]
ISSUE 2: ...
```

#### Orchestrator action on verdict
- **PASS** → orchestrator copies `/tmp/research_<topic>/plan_final.md` to `<output_dir>/plan.md` and begins the per-chapter loop.
- **FAIL** → inject the failure notes into the next round of Plan Generator prompts and return to Step 1.

#### ⚠️ Only the orchestrator may write to the output directory
No generator, evaluator, synthesizer, or verifier agent may write files to `output_dir`. The orchestrator is the sole agent that copies verified artifacts from `/tmp` into the project.

---

## Phase 1 — Per-Chapter Deep Planning

For each chapter, the orchestrator first runs a deep-planning round to produce the best possible initial content. Only after this round does the normal B → C quality-control loop begin. Use the **same N** determined in Step 0 for every chapter.

### Step 1: N Chapter Generators (Parallel)

Spawn N Chapter Generator agents **simultaneously**. Each receives:
- The verified `plan.md` (from `output_dir`)
- The chapter spec (title, file list, bullet content from `plan.md`)
- The content of all previously completed chapters (read from `output_dir`) for cross-chapter awareness
- Any failure notes from a previous Chapter Verifier round for this chapter

Each agent produces an **independent** version of the chapter without being aware of the others.

#### ⚠️ File Location Rule
Chapter Generators MUST write to `/tmp`, NOT the output directory:
```
/tmp/research_<topic>/ch<N>_v<1..N>/
  ├── index.md
  ├── <topic_a>.md
  └── <topic_b>.md
  (etc. — per plan spec)
```

Each chapter version MUST:
- Cover every file and bullet point listed in the plan spec for this chapter
- Include correct navigation footers on every content file
- Use LaTeX for all mathematical expressions (see Rules)
- Not duplicate content already covered in completed chapters

### Step 2: N Chapter Evaluators (Parallel)

Spawn N Chapter Evaluator agents **simultaneously**, one per candidate chapter. Each receives all N candidate chapter directories (from `/tmp`) and their assigned version index.

#### ⚠️ File Location Rule
Chapter Evaluators MUST write to `/tmp`, NOT the output directory:
```
/tmp/research_<topic>/eval_ch<N>_v<1..N>.md
```

Each evaluation MUST include:
- **Assigned version summary** (one paragraph)
- **Coverage**: does it address every bullet point in the plan spec?
- **Technical accuracy**: factual errors or unclear explanations
- **Clarity and depth**: appropriate for the stated audience
- **Cross-chapter hygiene**: does it avoid duplicating content from prior chapters?
- **Comparison to other versions**: what other versions do better or worse
- **Recommendation**: Accept / Accept with modifications / Reject (with reason)

### Step 3: 1 Chapter Synthesizer

Spawn **one** Chapter Synthesizer with all N candidate chapter directories and all N evaluations (read from `/tmp`).

#### ⚠️ File Location Rule
The Chapter Synthesizer MUST write the candidate final chapter to `/tmp`, NOT the output directory:
```
/tmp/research_<topic>/ch<N>_final/
```

The Chapter Synthesizer MUST:
1. Review all versions and evaluations
2. Select or construct the best version
3. Include a `synthesis_notes.md` in the output directory explaining what was chosen and why

### Step 4: 1 Chapter Verifier

Spawn **one** Chapter Verifier with:
- The candidate final chapter (from `/tmp/research_<topic>/ch<N>_final/`)
- The chapter spec from `plan.md`
- Content of all previously completed chapters (read from `output_dir`)
- Any prior failure notes for this chapter

The Chapter Verifier MUST:
1. Confirm every plan bullet point is covered
2. Check for cross-chapter duplication with completed chapters
3. Check navigation footers are present on every content file
4. Check LaTeX formatting rules are followed
5. Return a verdict:

**VERDICT: PASS** — chapter is ready for B/C quality control.

**VERDICT: FAIL** — list each issue precisely:
```
ISSUE 1: [Description of gap or flaw]
ISSUE 2: ...
```

#### Orchestrator action on verdict
- **PASS** → orchestrator copies `/tmp/research_<topic>/ch<N>_final/` (excluding `synthesis_notes.md`) to `<output_dir>/ch<N>_<title>/` and begins the normal B → C loop on the copied files.
- **FAIL** → inject the failure notes into the next round of Chapter Generator prompts and return to Step 1 for this chapter.

#### ⚠️ Only the orchestrator may write to the output directory
No generator, evaluator, synthesizer, or verifier agent may write files to `output_dir`. The orchestrator copies the verified chapter from `/tmp` into the project on PASS.

---

## Agents

**Critical rule**: Each agent invocation is a fresh sub-agent with no memory of prior invocations. The orchestrator passes context explicitly via the prompt. Agents B and C have never seen the content they are reviewing — they approach it with zero attachment.

### Plan Generator
- Produces one independent candidate structural plan for the topic.
- Writes to `/tmp/research_<topic>/plan_v<K>.md`.
- Never aware of other Plan Generators' output.

### Plan Evaluator
- Reads all N candidate plans from `/tmp`.
- Writes a critique and recommendation to `/tmp/research_<topic>/eval_plan_v<K>.md`.
- Never modifies plan files.

### Plan Synthesizer
- Reads all N plans and evaluations from `/tmp`.
- Writes the best (or hybrid) plan to `/tmp/research_<topic>/plan_final.md`.
- Includes a Selection Rationale section.

### Plan Verifier
- Reads the candidate final plan from `/tmp`.
- Returns PASS or FAIL with specific issues.
- Never modifies plan files.

### Chapter Generator
- Produces one independent candidate chapter version.
- Writes to `/tmp/research_<topic>/ch<N>_v<K>/`.
- Never aware of other Chapter Generators' output.

### Chapter Evaluator
- Reads all N candidate chapter directories from `/tmp`.
- Writes a critique to `/tmp/research_<topic>/eval_ch<N>_v<K>.md`.
- Never modifies chapter files.

### Chapter Synthesizer
- Reads all N chapter versions and evaluations from `/tmp`.
- Writes the best (or hybrid) chapter to `/tmp/research_<topic>/ch<N>_final/`.
- Writes `synthesis_notes.md` explaining the selection.

### Chapter Verifier
- Reads the candidate final chapter from `/tmp`.
- Returns PASS or FAIL with specific issues.
- Never modifies chapter files.

### Agent A — Generator
- The **only agent that writes or modifies guide content in `output_dir`**.
- Runs after the deep-planning round completes and the chapter is copied to `output_dir`.
- When revising, applies feedback verbatim — does not silently skip or partially apply suggestions.
- After applying feedback, writes a brief **change log** note at the bottom of the relevant `compression_analysis.md` confirming what was done.
- Receives from orchestrator: the plan, the list of files to write, and any feedback to apply.

### Agent B — Critic
- A **separate agent invocation** that has never seen this content before. It reads as a cold, independent reviewer.
- **Reads** chapter files and evaluates them. Never modifies content files.
- **Scope (strict):** Agent B flags only issues where a downstream reader would (a) get a wrong numerical answer, (b) implement something incorrectly, or (c) be materially misled into a wrong conceptual understanding. Agent B **must NOT flag**: style preferences, sentence structure choices, abbreviation consistency, cross-reference formatting conventions, prose wordiness, or any issue a reader can resolve with basic context. If in doubt, do not flag it.
- Checks only:
  1. **Factual correctness** — is a stated fact, formula, or derivation wrong? Would following it produce incorrect code or a wrong result?
  2. **Critical coherence** — is a concept used before it is defined in a way that blocks comprehension (not just mildly inconvenient)?
  3. **Critical structural gaps** — is a planned file missing, or does a file omit content that a later chapter explicitly depends on?
- **Maximum 5 items per pass.** If more than 5 genuine correctness issues exist, list only the 5 most severe.
- Writes feedback as a numbered list (≤5 items). Each item: file, approximate line, the specific correctness error, and a concrete fix.
- When there are no correctness issues, writes exactly: **"No feedback — chapter approved."**
- Writes output to `<chapter_dir>/b_review.md` (append, with pass number heading). Does NOT write to `compression_analysis.md`.
- Receives from orchestrator: the plan, the list of chapter files to read (by path), and any prior feedback from previous passes.

### Agent C — Compressor
- A **separate agent invocation** that has never seen this content before. It reads as a pure content editor whose sole job is to find and eliminate redundancy and bloat.
- **Reads** chapter files and identifies redundancy and bloat. Never modifies content files.
- **Scope (strict):** Agent C looks ONLY for: duplicate explanations of the same concept across files, restated tables, verbose prose that can be shortened without losing meaning, over-long code comments that restate what the code already shows, repeated examples, and hedging language that adds no value. Agent C **must NOT** flag factual errors, missing information, or correctness issues — that is Agent B's job exclusively.
- **Must find something to compress.** Agent C is not done until it has read every line and made an honest attempt to identify redundancy. A verdict of `Crucial updates: no` is only valid if Agent C has populated both a `## Load-Bearing Evidence` section (see format below) AND at least one `## MINOR Suggestions` item. A compression_analysis.md with an empty MINOR section and no Load-Bearing Evidence section will be **rejected by the orchestrator** and Agent C will be re-spawned.
- If a chapter truly has zero CRUCIAL bloat (uncommon), Agent C must:
  1. Populate `## Load-Bearing Evidence` with a bullet per file quoting a specific line or passage and explaining why it cannot be cut.
  2. Provide at least one MINOR suggestion — verbose phrasing, redundant adjectives, over-long code comments, restated variable names in prose. There is always something.
  3. Only then may VERDICT be `Crucial updates: no`.
- Writes its output to `<chapter_dir>/compression_analysis.md`, structured as:

```
# Compression Analysis: <Chapter Title> — Pass <N>

## Summary
- Total files analyzed: <N>
- Estimated current line count: ~<X> lines
- Estimated post-compression line count: ~<Y> lines
- Estimated reduction: ~<Z>%

## CRUCIAL Suggestions
### [<filename>] ~lines <range>
**Issue:** <description of redundancy or bloat>
**Suggestion:** <specific action to take>

## MINOR Suggestions
### [<filename>] ~lines <range>
**Issue:** ...
**Suggestion:** ...

## Load-Bearing Evidence
(Required when VERDICT is "Crucial updates: no". One bullet per file quoting a specific line/passage and explaining why it cannot be cut. Omitting this section invalidates the verdict.)
- `<filename>` line ~<N>: "<quoted text>" — load-bearing because <reason>

## VERDICT
- Crucial updates: yes | no
```

- **CRUCIAL**: Significant redundancy (duplicate sections, restated tables, over-explained concepts already covered elsewhere in the same or prior chapters). Agent A must address all CRUCIAL items.
- **MINOR**: Verbose phrasing, low-value elaboration, repeated code boilerplate, hedging language, redundant inline comments. Agent A may address MINOR items at discretion.
- When `Crucial updates: no`, Agent C's pass is complete for that scope — but only if the format rules above are satisfied.
- On each subsequent pass, Agent C increments the pass number and only re-checks items from the previous pass that were flagged as CRUCIAL.
- Receives from orchestrator: the list of chapter files to read (by path), the current pass number, and (on re-check passes) the prior compression_analysis.md.

**Orchestrator enforcement**: After receiving Agent C's output, the orchestrator must verify:
1. The `## Summary` section has non-zero line count estimates.
2. If VERDICT is `Crucial updates: no`: `## Load-Bearing Evidence` is present and non-empty, AND `## MINOR Suggestions` is non-empty.
3. If either check fails, re-spawn Agent C with the instruction: "Your previous compression_analysis.md was rejected. Reason: [specific missing section]. Re-read all files and produce a valid analysis."

---

## Per-Chapter Loop (After Deep-Planning Round)

Once the orchestrator has copied the verified chapter to `<output_dir>/ch<N>_<title>/`, run the standard quality-control loop:

```
repeat:
    repeat:
        [Orchestrator spawns Agent B — fresh invocation, no shared context with A]
            Agent B reads the chapter and produces feedback
        if feedback is "No feedback — chapter approved":
            break
        [Orchestrator spawns Agent A — fresh invocation, passes B's feedback explicitly]
            Agent A applies all feedback
    [Orchestrator spawns Agent C — fresh invocation, no shared context with A or B]
        Agent C reads the chapter and produces compression_analysis.md (or updates it)
    if compression_analysis VERDICT is "Crucial updates: no":
        break
    [Orchestrator spawns Agent A — fresh invocation, passes C's CRUCIAL suggestions explicitly]
        Agent A applies all CRUCIAL compression suggestions
until Agent B approves AND Agent C verdict is "Crucial updates: no"
```

A chapter is **done** only when both conditions hold simultaneously in the same iteration.

**Before accepting Agent C's verdict**, the orchestrator must check:
- `## Summary` has non-zero line counts.
- If VERDICT is `Crucial updates: no`: `## Load-Bearing Evidence` is present and non-empty, AND `## MINOR Suggestions` has at least one item.
- If either check fails, re-spawn Agent C (directly, as orchestrator) with explicit rejection reason. Do not accept a `Crucial updates: no` verdict without this evidence.

---

## Final Pass — Index and Cross-Chapter Coherence

After all chapters are complete:

1. **Orchestrator spawns Agent A** to write `<output_dir>/index.md`. The index must contain:
   - A 1–2 sentence description of the guide (scope and audience).
   - A **"How to Use This Guide"** table: common reader goals mapped to recommended chapter paths and direct deep links.
   - A **Chapter Index** table: chapter number, title, one-line description, key operations or concepts. **Every chapter entry in this table must be a clickable markdown link to that chapter's `index.md`**, e.g. `[Ch 1 — Title](ch1_title/index.md)`.
   - A **Quick Reference** table: the most-used API calls or concepts, what each does, and where to learn more.
   - A **Prerequisites** section.
   - A **Source Code Location** section (if applicable).

2. Run the full A→B→C loop on the index and all chapters together:
   - **Orchestrator spawns Agent B** (fresh) to check cross-chapter consistency (terminology, notation, forward/back references).
   - **Orchestrator spawns Agent C** (fresh) to write `<output_dir>/compression_analysis.md` covering cross-chapter redundancy (duplicate tables, concepts defined in multiple chapters verbatim, etc.).
   - **Orchestrator spawns Agent A** (fresh) to apply all feedback and CRUCIAL compression suggestions.
   - Repeat until Agent B produces **"No feedback — guide approved."** and Agent C produces **"Crucial updates: no"** for both the cross-chapter analysis and each chapter's own analysis.

---

## Completion Condition

The process is complete when **all** of the following are true:

- Every chapter's `compression_analysis.md` ends with `Crucial updates: no`.
- `<output_dir>/compression_analysis.md` (cross-chapter) ends with `Crucial updates: no`.
- Agent B's last pass over the full guide produces **"No feedback — guide approved."**

At that point, report back to the user with:
- The output directory path.
- N chosen and difficulty rationale.
- A summary of what was generated (chapter count, file count, rough line count).
- Any plan amendments that were made during the process and why.

---

## File Layout Convention

```
/tmp/research_<topic>/           # ALL intermediate artifacts — never committed
├── plan_v1.md                   # Candidate plans from Plan Generators
├── plan_v2.md
├── ...
├── eval_plan_v1.md              # Plan evaluations
├── eval_plan_v2.md
├── ...
├── plan_final.md                # Synthesized plan (copied to output_dir/plan.md on PASS)
├── ch1_v1/                      # Candidate chapter 1 versions
├── ch1_v2/
├── ...
├── eval_ch1_v1.md               # Chapter 1 evaluations
├── eval_ch1_v2.md
├── ...
└── ch1_final/                   # Synthesized chapter 1 (copied to output_dir on PASS)

<output_dir>/                    # ONLY orchestrator-verified artifacts land here
├── plan.md                        # Initial plan (updated if amended)
├── index.md                       # Top-level guide index (written in final pass)
├── compression_analysis.md        # Cross-chapter compression analysis
├── ch1_<title>/
│   ├── index.md
│   ├── <topic_a>.md
│   ├── <topic_b>.md
│   ├── b_review.md
│   └── compression_analysis.md
├── ch2_<title>/
│   └── ...
└── ...
```

Chapter directory names use the format `ch<N>_<short_snake_case_title>`.

---

## Rules

1. Agent A is the only agent that writes files to `output_dir`. Agents B and C only read and produce feedback.
2. No generator, evaluator, synthesizer, or verifier agent may write to `output_dir`. Only the orchestrator copies verified artifacts from `/tmp`.
3. Every CRUCIAL suggestion from Agent B or Agent C must be applied before moving forward.
4. The plan in `plan.md` is the authority on structure. If content contradicts the plan, Agent B flags it; Agent A fixes the content or proposes a plan amendment.
5. Do not move to the next chapter until the current chapter satisfies the completion condition (B approved + C `Crucial updates: no`).
6. Do not write the index until all chapters are done.
7. Compression analysis files are append-only across passes (each pass adds a new dated section). Never delete a previous pass's analysis.
8. Keep `index.md` and chapter `index.md` files as pure navigation — no content that belongs in a section file.
9. All cross-chapter references use relative markdown links.
   - **Chapter `index.md` files must use clickable markdown links for every file reference** — both in navigation tables and in reading-order lists. Plain backtick filenames (e.g. `` `topic.md` ``) are not acceptable; every reference must be in the form `[`topic.md`](./topic.md)`.
   - **Every content file** (any `.md` that is not `index.md`, `b_review.md`, `compression_analysis.md`, or `plan.md`) **must end with a navigation footer**:
     - If it is not the last file in the chapter: `---\n\n**Next:** [\`next_file.md\`](./next_file.md)`
     - If it is the last file in the chapter but not the last chapter: `---\n\n**Next:** [Chapter N+1 — Title](../chN+1_title/index.md)`
     - If it is the last file of the entire guide's last chapter: `---\n\n**End of guide.** Return to [Guide Index](../index.md)`
   - Agent A is responsible for adding these footers when writing each file. Agent B must flag any content file that is missing its navigation footer as a structural gap.
10. **Agent B and Agent C are always separate agent invocations.** The orchestrator must never allow B or C to run in the same context as A. This isolation is what makes the review adversarial and trustworthy. Agent B writes to `b_review.md`; Agent C writes to `compression_analysis.md`. These are separate files.
11. **Agent C must produce a Summary section with actual line count estimates.** A verdict of `Crucial updates: no` without a populated Summary section is invalid.
12. **A `Crucial updates: no` verdict requires both `## Load-Bearing Evidence` (non-empty, one bullet per file with a quoted line) and at least one `## MINOR Suggestions` item.** The orchestrator must reject and re-spawn Agent C if either is absent. There is no such thing as a chapter with zero MINOR issues.
13. **The orchestrator must never rubber-stamp Agent C's verdict.** It must explicitly verify the two conditions in rule 12 before accepting the verdict and advancing.
14. **Every chapter `index.md` must use clickable markdown links for all file references.** Agent A must write all file references in navigation tables and reading-order lists as `[`filename.md`](./filename.md)`, never as plain backtick names. Agent B must flag plain backtick-only file references as a structural gap.
15. **Every content file must end with the correct navigation footer** (see rule 9). Agent A is responsible for adding it when writing the file. Agent B must flag missing footers as a structural gap.
16. **The guide-level `index.md` chapter table must use clickable links** to each chapter's `index.md`. Entries of the form `Ch N — Title` without a hyperlink are not acceptable.
17. **All mathematical equations must use LaTeX formatting.**
    - Display (block) equations: `$$...$$` on their own lines, or ` ```math ` fenced blocks.
    - Inline expressions: `$...$`.
    - Shape annotations, arithmetic calculations, and pseudocode: fenced code blocks or plain text.
    - Agent B must flag any display equation written in plain text or a code block as a structural gap.
    - **Never use underscores inside `\text{...}`** — use spaces instead: `\text{expert capacity}` not `\text{expert\_capacity}`.
    - **Never use `\texttt{name_with_underscores}` in `$$...$$` or `$...$` blocks.** Use ` ```math ` fenced blocks for display equations containing `\texttt` with underscores.
    - **Never use `\!` in `$$...$$` blocks.** Use ` ```math ` fenced blocks instead.
    - **Summary**: Use ` ```math ` when the equation contains `\texttt` with underscores or `\!`. Use `$$...$$` for all other display equations.
18. **N is fixed for the entire topic at Step 0 and never re-assessed.** The same N applies to Phase 0 and every chapter's deep-planning round.
19. **The `synthesis_notes.md` file written by the Chapter Synthesizer is never copied to `output_dir`.** It is a `/tmp`-only artifact.
