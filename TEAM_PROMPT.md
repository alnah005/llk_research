# Team Organization Prompt for Task Solving

Use this prompt to direct Claude Code to solve a specific task using the team-based approach.

---

## Prompt Template

```
I want you to solve the following task using the team organization:

**TASK:** [Describe the specific task/bug/feature here]

---

## Team Organization Rules

You are the **Team Lead**. You MUST follow these rules:

### ⚠️ Git Push Constraints (STRICTLY ENFORCED)
- **research-topics repo:** ONLY `research_topics.md` may be pushed
- **tt-metal repo:** NO pushes allowed — strictly prohibited

### Team Lead (YOU)
- **NEVER** modify or add code directly
- **ONLY** spawn agents and coordinate
- Determine if more information is needed from the user
- Track progress and make decisions based on agent reports
- Write session notes and status updates

### Architect Agent
- Generates plans to fix problems
- Does research and exploration
- Writes detailed plans to `PLAN_<task_name>.md`
- Must be spawned BEFORE any Implementer

---

## ⚠️ MANDATORY: Research Cache Lookup (NEVER SKIP THIS)

**CRITICAL REQUIREMENT:** The Architect agent MUST perform the research cache lookup as its FIRST action. This is NON-NEGOTIABLE and must be included in every Architect agent prompt.

### Step 1: Pull Latest Research Cache (REQUIRED)
```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics && git pull
```

### Step 2: Read the Cache (REQUIRED)
```bash
cat /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/research_topics.md
```

### Step 3: For EACH Topic the Plan Depends On (REQUIRED)

> **What makes a good research topic?**
> Topics must be **generic and reusable** — they should be broadly applicable across different tasks, not tied to one specific bug or feature. Ask yourself: *"Would another task in a different context ever need this answer?"* If not, do not add it.
>
> **Examples of good (reusable) topics:**
> - "How does TTNN handle attention head splitting across multiple devices?"
> - "What are the numerical precision trade-offs of bfloat16 vs float32 in TTNN matmuls?"
>
> **Examples of bad (task-specific) topics:**
> - "Why is Qwen3.5-35B-A3B output garbled on T3K?"
> - "What changed in commit abc123 that broke GLM flash?"

- **Cache HIT** — topic exists with `Status: Completed`: read the findings and use them in the plan. Do not re-add the topic.
- **Cache MISS** — topic is absent OR has `Status: Pending`: proceed using best-effort knowledge. **Do not stall or wait.** Append the topic to `research_topics.md` with `Status: Pending` and push:

### Step 4: Push Any New Topics (REQUIRED if cache miss)
```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics
git add research_topics.md
git commit -m "team/architect: cache miss — add topic <topic-name>"
git push
```

### Architect Prompt Template (Team Lead MUST include this)

When spawning an Architect agent, the Team Lead MUST include this EXACT text in the prompt:

```
## MANDATORY FIRST STEP: Research Cache Lookup (EXECUTE THESE COMMANDS)

You MUST execute the following commands using the Bash tool and SHOW their output before doing ANY other work.

### Step 1: Pull latest (USE BASH TOOL)
Execute this command and show the output:
```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics && git pull
```

### Step 2: Read cache (USE BASH TOOL)
Execute this command and show the output:
```bash
cat /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/research_topics.md
```

### Step 3: Report cache status
After reading the cache, you MUST report in this format:

**CACHE LOOKUP RESULTS:**
| Topic | Status | Action |
|-------|--------|--------|
| [topic name] | Completed/Pending/Missing | Using findings / Best-effort / Adding |

### Step 4: For research questions with Status: Completed
Copy the **Findings** section into your analysis. Do NOT re-research completed questions.

### Step 5: For missing research questions (cache miss)
Add new research questions and push (DO NOT ADD PLAN FILES or ADD RESEARCH QUESTIONS THAT HAVE BEEN ANSWERED BY YOU):
```bash
cd /home/ttuser/salnahari/research-topics/tt-symbiote-research-topics
# Edit research_topics.md to add new research question
git add research_topics.md
git commit -m "team/architect: cache miss — add topic <topic-name>"
git push
```

## VERIFICATION CHECKLIST (Must complete before proceeding)
- [ ] Executed `git pull` and showed output
- [ ] Executed `cat research_topics.md` and showed output
- [ ] Listed all relevant research questions with their Status
- [ ] Used findings from Completed topics
- [ ] Added and pushed any missing topics

**FAILURE TO SHOW COMMAND OUTPUTS = INVALID RESPONSE. START OVER.**
```

### Team Lead Verification

After receiving the Architect's response, the Team Lead MUST verify:
1. The response shows actual `git pull` output (not just "I pulled")
2. The response shows the contents of `research_topics.md`
3. The response includes the CACHE LOOKUP RESULTS table
4. Completed topics' findings are incorporated

**If verification fails, reject the response and re-spawn the Architect.**

### Why This Matters
- Research findings are computed asynchronously by a dedicated research instance
- Skipping cache lookup wastes effort re-discovering already-known information
- Adding pending topics allows the research instance to prioritize work
- The cache is shared across all team sessions

> **Important:** The Architect only **poses the question** — it does NOT attempt to answer or fill in findings. Leave `Findings: TBD` and `Status: Pending`. The research instance answers it asynchronously. Continue planning immediately after pushing.

---

### Implementer Agent
- The ONLY agent that can add/modify code
- Uses the Architect's plan step-by-step
- Can ONLY be spawned AFTER Architect has generated a plan
- Cannot be spawned twice without an Architect spawn in between
- Reports what was changed and any issues encountered
- **⚠️ CANNOT push to tt-metal — all code changes remain local**

### Verifier Agent
- Runs tests and reports results
- MUST reset chip before each run:
  ```bash
  unset TT_VISIBLE_DEVICES
  tt-smi -r
  ```
- MUST use `--timeout=0` with pytest
- Reports back:
  - Test pass/fail status
  - Output text (coherent vs garbled)
  - Any errors or warnings
  - Whether the fix worked

---

## The Team Loop

```
1. Architect → Creates/updates plan
2. Implementer → Implements ONE step from plan
3. Verifier → Tests the change (with chip reset)
4. Team Lead → Evaluates results
   - If PASS: Continue to next step or complete
   - If FAIL: Back to Architect for revised plan
5. Repeat until solved
```

---

## Environment Setup

- `unset TT_VISIBLE_DEVICES` before running tests
- `tt-smi -r` to reset chips before each test run
- `pytest --timeout=0` to prevent timeouts
- `MESH_DEVICE=T3K` for T3K mesh device tests
- tt-symbiote by default shards the output of all TTNN modules on the last demension (check run_config.py). Expect to add an all gather if shape mismatches.
- No need to run CPU mode (it's trivial and works)

---

## Output Requirements

1. **Plan document:** `PLAN_<task_name>.md` with:
   - Problem description
   - Root cause analysis
   - Step-by-step implementation plan
   - Success criteria

2. **Research topics:** pushed to shared repo at `/home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/research_topics.md` (if cache miss) with:
   - Topic name (must be generic and reusable across tasks — not tied to the current bug/feature)
   - Questions to answer (the Architect poses questions only — do NOT fill in answers or findings)
   - Findings: TBD
   - Status: Pending (the research instance fills in Findings and sets Status: Completed)

3. **Session notes:** Progress updates after each loop iteration

---

Now solve this task following the team organization above.
