# Prompts Library

> Save every prompt that works. Label it. Reuse forever.

---

## PROMPT-001 — Session Opener (Strategy Chat)
Use this to open every new Claude AI advisor session.

I am working on this research paper:

Title: "Does Retrieval Actually Help? A Large-Scale Empirical Study of When RAG Succeeds, Fails, and Wastes Money"

Type: Measurement study. NOT a systems paper.

Primary contribution: Systematic measurement of when retrieval helps, hurts, or is neutral across NQ, HotpotQA, and MuSiQue using GPT-4o and GPT-4o-mini. Releasing labeled dataset on HuggingFace and public leaderboard.

What we do NOT claim:
- We do not claim our system beats baselines
- We do not claim cost reduction percentages
- This is not a systems paper

Current status: Building FAISS index locally on M1 Pro. Next step: label generation.

My question today: [INSERT YOUR QUESTION]

---

## PROMPT-002 — Start Any Claude Code Session (Implementation)
**Use for:** Opening Claude Code for a coding session
**Paste this:**
```
Here is my project context:
[paste MASTER_CONTEXT.md]

Here is today's task:
[paste CURRENT_TASK.md]

Read these carefully. Then help me implement exactly what is described in CURRENT_TASK.md.
Do not add features beyond what is listed. Do not refactor code I haven't asked you to touch.
```

---

## PROMPT-003 — Get Claude Code Prompt From Strategy Session
**Use for:** End of a strategy session — ask AI to hand off to Claude Code
**Paste this:**
```
Based on everything we just decided, write me a complete CURRENT_TASK.md
that I can paste at the start of my next Claude Code session.
Be specific about files, what done means, and constraints.
```

---

## PROMPT-005 — IEEE Reviewer Simulation
**Use for:** Getting AI to review your paper draft as a harsh IEEE reviewer
**Paste this:**
```
Act as a senior IEEE reviewer with expertise in information retrieval and NLP.
Review the following paper section. Be harsh. List every weakness that would cause rejection.
Specifically check for: reproducibility gaps, unfair baselines, missing statistical tests,
overclaimed contributions, and insufficient ablations.

[paste paper section]
```
