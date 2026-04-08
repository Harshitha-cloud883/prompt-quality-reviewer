"""
Prompt Quality Reviewer - OpenEnv Environment
A real-world environment for training AI agents to evaluate and improve prompt quality.
"""

import re
import json
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Typed Models ────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Agent action: submit a quality score and optional improved prompt."""
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score between 0.0 and 1.0")
    feedback: str = Field(..., description="Explanation of the score")
    improved_prompt: Optional[str] = Field(None, description="Optional improved version of the prompt")


class Observation(BaseModel):
    """What the agent sees at each step."""
    prompt: str = Field(..., description="The prompt to evaluate")
    task_description: str = Field(..., description="What the prompt is trying to accomplish")
    difficulty: str = Field(..., description="Task difficulty: easy | medium | hard")
    step_number: int = Field(..., description="Current step (1-indexed)")
    max_steps: int = Field(..., description="Maximum steps allowed")
    previous_feedback: Optional[str] = Field(None, description="Feedback from previous step if any")


class StepResult(BaseModel):
    """Result returned after each step."""
    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict


class EnvironmentState(BaseModel):
    """Full serializable environment state."""
    current_task_index: int
    step_number: int
    total_reward: float
    done: bool
    tasks_completed: int
    last_feedback: Optional[str]


# ─── Task Bank ────────────────────────────────────────────────────────────────

TASKS = [
    # ── EASY ──────────────────────────────────────────────────────────────────
    {
        "difficulty": "easy",
        "task_description": "Write a prompt to ask an AI to summarize a news article.",
        "prompt": "summarize this",
        "ground_truth_score": 0.15,
        "issues": ["too vague", "no context", "no format requested", "no length constraint"],
        "ideal_improved": "Please summarize the following news article in 3 bullet points, focusing on the key facts, people involved, and outcome:\n\n[ARTICLE TEXT]",
    },
    {
        "difficulty": "easy",
        "task_description": "Write a prompt to get a recipe from an AI.",
        "prompt": "give me a recipe",
        "ground_truth_score": 0.10,
        "issues": ["no dish specified", "no dietary constraints", "no serving size", "no format"],
        "ideal_improved": "Give me a simple vegetarian pasta recipe for 2 people. Include ingredients with measurements, prep time, and step-by-step instructions.",
    },
    {
        "difficulty": "easy",
        "task_description": "Ask an AI to help debug Python code.",
        "prompt": "my code doesn't work fix it",
        "ground_truth_score": 0.05,
        "issues": ["no code provided", "no error message", "no context", "no expected behavior stated"],
        "ideal_improved": "I have a Python function that should return the sum of a list but it returns None. Here's the code:\n\n```python\ndef sum_list(lst):\n    total = 0\n    for x in lst:\n        total + x\n```\n\nWhat's wrong and how do I fix it?",
    },

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
        "difficulty": "medium",
        "task_description": "Write a prompt to generate a professional LinkedIn post about a new job.",
        "prompt": "Write a LinkedIn post about my new job at Google as a software engineer. Make it good.",
        "ground_truth_score": 0.45,
        "issues": ["'make it good' is vague", "no tone specified", "no length", "no personal details to include"],
        "ideal_improved": "Write a professional yet warm LinkedIn post announcing my new role as a Software Engineer at Google. Keep it to 3 short paragraphs: (1) excitement about the role, (2) gratitude to previous colleagues, (3) what I'm looking forward to. Avoid clichés. Tone: authentic, humble, enthusiastic.",
    },
    {
        "difficulty": "medium",
        "task_description": "Create a prompt to generate a study plan for an exam.",
        "prompt": "Create a study plan for my data structures exam next week.",
        "ground_truth_score": 0.40,
        "issues": ["no topics listed", "no available time per day", "no current knowledge level", "no preferred study format"],
        "ideal_improved": "Create a 7-day study plan for a Data Structures exam. Topics to cover: arrays, linked lists, stacks, queues, trees, graphs, and sorting algorithms. I can study 2 hours/day. I'm comfortable with arrays and stacks but weak on trees and graphs. Format it as a daily table with topics, resources, and practice problems.",
    },
    {
        "difficulty": "medium",
        "task_description": "Write a prompt to get an AI to write a cover letter.",
        "prompt": "Write a cover letter for a software developer job. I have 2 years of experience.",
        "ground_truth_score": 0.50,
        "issues": ["company name missing", "job description not included", "skills not listed", "no tone or length guidance"],
        "ideal_improved": "Write a concise, professional cover letter (under 300 words) for a Junior Software Developer role at a fintech startup. I have 2 years of experience in Python and React, built 3 production web apps, and I'm passionate about financial inclusion. Match tone to a modern startup. Highlight problem-solving and collaboration skills.",
    },

    # ── HARD ──────────────────────────────────────────────────────────────────
    {
        "difficulty": "hard",
        "task_description": "Design a system prompt for a customer support AI chatbot for an e-commerce platform.",
        "prompt": "You are a helpful customer support agent. Help customers with their issues.",
        "ground_truth_score": 0.30,
        "issues": [
            "no company identity", "no tone guidelines", "no escalation policy",
            "no scope of topics", "no prohibited actions", "no format instructions",
            "no handling of edge cases like refunds or complaints"
        ],
        "ideal_improved": (
            "You are Aria, a friendly and efficient customer support agent for ShopEase, an online retail platform.\n\n"
            "RESPONSIBILITIES:\n- Answer questions about orders, shipping, returns, and account issues\n- Process refund requests following the 30-day return policy\n- Escalate complaints involving fraud or legal matters to human agents\n\n"
            "TONE: Warm, professional, concise. Never use jargon. Always empathize first.\n\n"
            "LIMITS: Do not make promises about delivery dates. Do not access payment information.\n\n"
            "FORMAT: Respond in short paragraphs. Use bullet points for step-by-step instructions."
        ),
    },
    {
        "difficulty": "hard",
        "task_description": "Write a few-shot prompt to classify customer reviews as positive, negative, or neutral.",
        "prompt": "Classify these reviews as positive, negative, or neutral:\n1. Great product!\n2. It broke after a week.\n3. It's okay I guess.",
        "ground_truth_score": 0.55,
        "issues": [
            "no examples (few-shot) provided", "no output format specified",
            "no handling of ambiguous cases", "no confidence scoring", "no chain-of-thought"
        ],
        "ideal_improved": (
            "Classify each customer review as Positive, Negative, or Neutral. "
            "Return a JSON array with fields: review, label, confidence (0-1), reason.\n\n"
            "Examples:\n"
            "Review: 'Absolutely love this, works perfectly!' → Positive\n"
            "Review: 'Stopped working after 2 days, terrible quality.' → Negative\n"
            "Review: 'Package arrived. Item is as described.' → Neutral\n\n"
            "Now classify:\n1. Great product!\n2. It broke after a week.\n3. It's okay I guess."
        ),
    },
    {
        "difficulty": "hard",
        "task_description": "Write a meta-prompt to instruct an AI to generate other high-quality prompts.",
        "prompt": "Generate a good prompt for me about cooking.",
        "ground_truth_score": 0.20,
        "issues": [
            "no criteria for 'good'", "no structure requested", "no target audience",
            "no use case specified", "no evaluation criteria included"
        ],
        "ideal_improved": (
            "You are a prompt engineering expert. Generate a high-quality prompt for the following use case:\n\n"
            "USE CASE: Help a beginner home cook learn a new cuisine.\n\n"
            "Your prompt must include:\n"
            "1. A clear role/persona for the AI\n"
            "2. Specific task with constraints (e.g., skill level, time, ingredients)\n"
            "3. Desired output format (e.g., step-by-step, table, bullet list)\n"
            "4. At least one example of expected output\n"
            "5. Tone guidance\n\n"
            "Rate your own prompt on specificity (1-5), clarity (1-5), and completeness (1-5)."
        ),
    },
]


# ─── Reward Logic ─────────────────────────────────────────────────────────────

def compute_reward(action: Action, task: dict) -> tuple[float, dict]:
    """
    Compute a partial-progress reward based on:
    - Score accuracy (how close the agent's score is to ground truth)
    - Feedback quality (does feedback mention real issues?)
    - Improvement quality (if provided, is it meaningfully better?)
    """
    breakdown = {}

    # 1. Score accuracy (40% of reward)
    score_delta = abs(action.score - task["ground_truth_score"])
    score_reward = max(0.0, 1.0 - (score_delta / 0.5))  # within 0.5 = partial credit
    score_reward *= 0.40
    breakdown["score_accuracy"] = round(score_reward, 3)

    # 2. Feedback quality (35% of reward)
    feedback_lower = action.feedback.lower()
    issues_caught = sum(1 for issue in task["issues"] if any(word in feedback_lower for word in issue.split()))
    feedback_reward = min(1.0, issues_caught / max(1, len(task["issues"]))) * 0.35
    breakdown["feedback_quality"] = round(feedback_reward, 3)

    # 3. Improvement quality (25% of reward)
    improvement_reward = 0.0
    if action.improved_prompt:
        imp = action.improved_prompt
        ideal = task["ideal_improved"]

        # Heuristics: length, structure markers, specificity
        length_score = min(1.0, len(imp) / max(1, len(ideal)))
        has_structure = any(marker in imp for marker in [":", "\n", "-", "1.", "•", "["])
        specificity_score = min(1.0, len(imp.split()) / 30)

        improvement_reward = (length_score * 0.4 + (0.3 if has_structure else 0) + specificity_score * 0.3) * 0.25
    breakdown["improvement_quality"] = round(improvement_reward, 3)

    total = score_reward + feedback_reward + improvement_reward
    breakdown["total"] = round(total, 3)
    return round(total, 3), breakdown


# ─── Environment ──────────────────────────────────────────────────────────────

class PromptQualityEnv:
    """
    OpenEnv-compliant environment for evaluating and improving AI prompts.

    Tasks increase in difficulty: easy → medium → hard
    Agents score prompts (0.0–1.0), provide feedback, and optionally rewrite them.
    """

    MAX_STEPS_PER_TASK = 2  # agent gets 2 attempts per task
    TASKS = TASKS

    def __init__(self):
        self._task_index = 0
        self._step = 0
        self._total_reward = 0.0
        self._done = False
        self._tasks_completed = 0
        self._last_feedback = None

    def reset(self) -> Observation:
        """Reset environment to the first task."""
        self._task_index = 0
        self._step = 0
        self._total_reward = 0.0
        self._done = False
        self._tasks_completed = 0
        self._last_feedback = None
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Take one step: evaluate the action, compute reward, advance task."""
        if self._done:
            raise RuntimeError("Environment is done. Call reset() to start again.")

        task = self.TASKS[self._task_index]
        reward, breakdown = compute_reward(action, task)
        self._total_reward += reward
        self._step += 1
        self._last_feedback = action.feedback

        # Advance to next task after MAX_STEPS_PER_TASK or if agent scores well
        advance = (self._step >= self.MAX_STEPS_PER_TASK) or (reward >= 0.75)

        if advance:
            self._tasks_completed += 1
            self._task_index += 1
            self._step = 0
            self._last_feedback = None

        self._done = self._task_index >= len(self.TASKS)

        next_obs = self._make_observation() if not self._done else self._make_observation(done=True)

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=self._done,
            info={
                "reward_breakdown": breakdown,
                "tasks_completed": self._tasks_completed,
                "total_reward": round(self._total_reward, 3),
                "average_reward": round(self._total_reward / max(1, self._tasks_completed), 3),
            },
        )

    def state(self) -> EnvironmentState:
        """Return full serializable environment state."""
        return EnvironmentState(
            current_task_index=self._task_index,
            step_number=self._step,
            total_reward=round(self._total_reward, 3),
            done=self._done,
            tasks_completed=self._tasks_completed,
            last_feedback=self._last_feedback,
        )

    def _make_observation(self, done: bool = False) -> Observation:
        if done or self._task_index >= len(self.TASKS):
            # Return a terminal observation
            return Observation(
                prompt="[All tasks completed]",
                task_description="Environment finished.",
                difficulty="—",
                step_number=self._step,
                max_steps=self.MAX_STEPS_PER_TASK,
                previous_feedback=self._last_feedback,
            )
        task = self.TASKS[self._task_index]
        return Observation(
            prompt=task["prompt"],
            task_description=task["task_description"],
            difficulty=task["difficulty"],
            step_number=self._step + 1,
            max_steps=self.MAX_STEPS_PER_TASK,
            previous_feedback=self._last_feedback,
        )
