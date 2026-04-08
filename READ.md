# 🔍 Prompt Quality Reviewer — OpenEnv

A real-world OpenEnv-compliant environment where AI agents learn to evaluate and improve the quality of prompts. Agents score prompts, identify weaknesses, and optionally rewrite them — tasks span easy to hard difficulty.

---

## 🧠 Environment Description

AI agents are presented with poorly-written prompts and must:
1. **Score** the prompt quality (0.0 = terrible, 1.0 = perfect)
2. **Explain** what's wrong with the prompt (feedback)
3. **Rewrite** an improved version (optional, but rewarded)

This simulates a real-world skill: **prompt engineering quality assurance** — increasingly important as organizations deploy LLMs in production.

---

## 📊 Tasks

| # | Difficulty | Example Prompt | Core Issues |
|---|-----------|----------------|-------------|
| 1 | Easy | `"summarize this"` | No context, no format |
| 2 | Easy | `"give me a recipe"` | No dish, no constraints |
| 3 | Easy | `"my code doesn't work fix it"` | No code provided |
| 4 | Medium | `"Write a LinkedIn post... Make it good."` | Vague quality ask |
| 5 | Medium | `"Create a study plan for my exam"` | No topics, no schedule |
| 6 | Medium | `"Write a cover letter... I have 2 years experience"` | Missing job description |
| 7 | Hard | Basic customer support system prompt | Missing scope, escalation, tone |
| 8 | Hard | Review classification without few-shot | No examples, no format |
| 9 | Hard | `"Generate a good prompt about cooking"` | No evaluation criteria |

---

## 🎯 Action Space

```python
Action(
    score: float,           # 0.0–1.0 quality estimate (required)
    feedback: str,          # Issues identified (required)
    improved_prompt: str    # Rewritten prompt (optional)
)
```

## 👁 Observation Space

```python
Observation(
    prompt: str,             # Prompt to evaluate
    task_description: str,   # What the prompt is trying to do
    difficulty: str,         # "easy" | "medium" | "hard"
    step_number: int,        # Current step (max 2 per task)
    max_steps: int,
    previous_feedback: str   # Your last feedback (if any)
)
```

---

## 💰 Reward Function

Reward is **partial-progress** (0.0–1.0) with three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Score Accuracy | 40% | How close your score is to ground truth |
| Feedback Quality | 35% | How many real issues your feedback mentions |
| Improvement Quality | 25% | Quality of your rewritten prompt |

Early advancement: if reward ≥ 0.75, skip to next task immediately.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline agents
python baseline.py --agent all --seed 42

# Run the Gradio demo
python app.py

# Run grader on a custom agent
from grader import PromptQualityGrader
from environment import Action

def my_agent(obs):
    return Action(score=0.2, feedback="Too vague, missing context", improved_prompt=None)

grader = PromptQualityGrader()
print(grader.grade(my_agent))
```

---

## 📈 Baseline Scores (seed=42)

| Agent | Score | Description |
|-------|-------|-------------|
| Random | ~0.18 | Random scores, generic feedback |
| Heuristic | ~0.35 | Length + keyword rules |
| Oracle | ~0.55 | Near-optimal, difficulty-aware |

---

## 📁 File Structure

```
prompt_quality_env/
├── environment.py      # Core env: step(), reset(), state(), typed models
├── grader.py           # PromptQualityGrader — evaluates full agent runs
├── baseline.py         # 3 baseline agents with reproducible scores
├── app.py              # Gradio UI for Hugging Face Spaces
├── openenv.yaml        # OpenEnv spec
├── Dockerfile          # HF Spaces deployment
├── requirements.txt
└── README.md
```

---

## 🏗 OpenEnv Compliance

- ✅ Real-world task (not a game or toy)
- ✅ Typed models via Pydantic (`Action`, `Observation`, `StepResult`, `EnvironmentState`)
- ✅ `step()`, `reset()`, `state()` methods
- ✅ `openenv.yaml` with full spec
- ✅ Minimum 3 tasks with easy → medium → hard grading
- ✅ Meaningful reward with partial progress signals
- ✅ Baseline inference script with reproducible scores
- ✅ Gradio app + Dockerfile for Hugging Face Spaces deployment

---

## 👩‍💻 Author

Baikani Manirudh, Ambani Varshitha, Harshitha R
