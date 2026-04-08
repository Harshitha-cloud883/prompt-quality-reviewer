"""
inference.py — Prompt Quality Reviewer
OpenEnv × Meta × Scaler Hackathon

Follows the required inference.py spec:
- API_BASE_URL, MODEL_NAME, HF_TOKEN env variables
- All LLM calls via OpenAI client
- Stdout logs in START/STEP/END format
"""

import os
import json
from openai import OpenAI
from environment import PromptQualityEnv, Action

# ─── Environment Variables (required by checklist) ────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model-name>")
HF_TOKEN     = os.getenv("HF_TOKEN")  # No default — must be set externally

# Optional — only needed if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ─── OpenAI Client ────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key",
)


# ─── LLM-Powered Agent ────────────────────────────────────────────────────────

def llm_agent(obs) -> Action:
    """
    Calls the LLM via OpenAI client to evaluate the prompt.
    Returns a structured Action with score, feedback, and improved_prompt.
    """
    system_prompt = """You are an expert prompt engineer. Your job is to evaluate the quality of AI prompts.

Given a prompt and the task it's meant to accomplish, you must:
1. Score the prompt quality from 0.0 (terrible) to 1.0 (perfect)
2. Identify specific issues with the prompt
3. Provide an improved version

Respond ONLY in this exact JSON format:
{
  "score": <float 0.0-1.0>,
  "feedback": "<specific issues identified>",
  "improved_prompt": "<rewritten better prompt>"
}"""

    user_message = f"""Task the prompt should accomplish:
{obs.task_description}

Prompt to evaluate:
"{obs.prompt}"

Difficulty level: {obs.difficulty}
{f'Your previous feedback: {obs.previous_feedback}' if obs.previous_feedback else ''}

Evaluate this prompt and respond in the required JSON format."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        return Action(
            score=float(data.get("score", 0.3)),
            feedback=data.get("feedback", "No feedback provided."),
            improved_prompt=data.get("improved_prompt"),
        )
    except Exception:
        # Fallback if JSON parsing fails
        return Action(
            score=0.3,
            feedback=raw[:300] if raw else "Could not parse response.",
            improved_prompt=None,
        )


# ─── Main Inference Loop with START/STEP/END logging ─────────────────────────

def run_inference():
    env = PromptQualityEnv()
    obs = env.reset()

    # ── START ──
    print("START")
    print(json.dumps({
        "event": "start",
        "total_tasks": len(env.TASKS),
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
    }))

    step_num = 0

    while True:
        step_num += 1
        action = llm_agent(obs)
        result = env.step(action)

        # ── STEP ──
        print("STEP")
        print(json.dumps({
            "event": "step",
            "step": step_num,
            "difficulty": obs.difficulty,
            "prompt_evaluated": obs.prompt[:80],
            "agent_score": action.score,
            "reward": result.reward,
            "reward_breakdown": result.info.get("reward_breakdown", {}),
            "tasks_completed": result.info.get("tasks_completed", 0),
            "done": result.done,
        }))

        if result.done:
            break
        obs = result.observation

    state = env.state()

    # ── END ──
    print("END")
    print(json.dumps({
        "event": "end",
        "total_steps": step_num,
        "tasks_completed": state.tasks_completed,
        "total_reward": state.total_reward,
        "average_reward": round(state.total_reward / max(1, state.tasks_completed), 4),
        "passed": state.total_reward / max(1, state.tasks_completed) >= 0.50,
    }))


if __name__ == "__main__":
    run_inference()
