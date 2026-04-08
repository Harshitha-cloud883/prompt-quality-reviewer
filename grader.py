"""
Grader for the Prompt Quality Reviewer environment.
Evaluates agent performance across all 9 tasks and returns a final normalized score.
"""

from environment import PromptQualityEnv, Action, compute_reward


class PromptQualityGrader:
    """
    Grades an agent's full run through the Prompt Quality Reviewer environment.

    Scoring:
    - Runs the agent through all 9 tasks (easy → medium → hard)
    - Computes per-task reward using the three-component reward function
    - Returns average reward across all tasks as final score (0.0–1.0)
    """

    def __init__(self):
        self.env = PromptQualityEnv()

    def grade(self, agent_fn) -> dict:
        """
        Grade an agent function.

        Args:
            agent_fn: A callable that takes an Observation and returns an Action.

        Returns:
            dict with keys: total_score, per_task_scores, breakdown
        """
        obs = self.env.reset()
        per_task_scores = []
        breakdowns = []

        while True:
            action = agent_fn(obs)
            result = self.env.step(action)
            per_task_scores.append(result.reward)
            breakdowns.append(result.info.get("reward_breakdown", {}))

            if result.done:
                break
            obs = result.observation

        total_score = sum(per_task_scores) / len(per_task_scores)

        return {
            "total_score": round(total_score, 4),
            "per_task_scores": [round(s, 4) for s in per_task_scores],
            "num_tasks": len(per_task_scores),
            "breakdown_per_task": breakdowns,
            "passed": total_score >= 0.50,
        }


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def dummy_agent(obs):
        """A simple rule-based dummy agent for testing."""
        # Heuristic: short prompts = low score
        prompt_len = len(obs.prompt.split())
        score = min(0.9, max(0.05, prompt_len / 50))
        return Action(
            score=score,
            feedback=f"The prompt is {'very short and vague' if prompt_len < 10 else 'somewhat clear but missing constraints'}. "
                     f"Issues: lacks context, no format specified, no constraints given.",
            improved_prompt=obs.prompt + " Please provide a detailed, structured response with examples."
        )

    grader = PromptQualityGrader()
    results = grader.grade(dummy_agent)

    print("=" * 50)
    print("GRADER RESULTS")
    print("=" * 50)
    print(f"Total Score : {results['total_score']}")
    print(f"Passed      : {results['passed']}")
    print(f"Tasks Done  : {results['num_tasks']}")
    print()
    for i, (score, breakdown) in enumerate(zip(results["per_task_scores"], results["breakdown_per_task"])):
        print(f"Task {i+1:02d} → Score: {score} | {breakdown}")
