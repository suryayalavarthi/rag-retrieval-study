# Prompts Library

> Save every prompt that works. Label it. Reuse forever.

---

## PROMPT-001 — Start Any AI Session (Strategy Chat)
**Use for:** Opening any new conversation with Claude.ai or ChatGPT about this paper
**Paste this:**
```
I'm working on a research paper: "Retrieve, Assess, Route: Cost-Efficient RAG through Adaptive Context Sufficiency Scoring"

Here is my full project context:
[paste MASTER_CONTEXT.md]

Here is my current task:
[paste CURRENT_TASK.md]

Please read both carefully before responding. Ask clarifying questions if needed.
```

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

## PROMPT-004 — Second Verifier (Sufficiency Judge)
**Use for:** Claude API call to verify borderline sufficiency scores
**System prompt:**
```
You are a retrieval quality judge. Given a question and retrieved passages,
determine whether the passages contain sufficient information to answer the question correctly.
Respond with JSON only: {"sufficient": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}
Do not answer the question. Only judge whether the context is sufficient to answer it.
```
**User message format:**
```
Question: {query}

Retrieved passages:
{passage_1}
{passage_2}
{passage_3}

Is the retrieved context sufficient to answer the question?
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
