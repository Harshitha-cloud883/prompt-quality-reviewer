"""
Baseline Inference Script — Prompt Quality Reviewer
Runs three baseline agents and prints reproducible scores.

Agents:
  1. random_agent   — random scores, generic feedback
  2. heuristic_agent — length + keyword heuristics
  3. oracle_agent   — near-optimal (uses ground truth hints)

Usage:
  python baseline.py
  python baseline.py --agent heuristic
  python baseline.py --agent all --seed 42
"""

import argparse
import random
from environment import PromptQualityEnv, Action
from grader import PromptQualityGrader


# ─── Agent Definitions ────────────────────────────────────────────────────────

def random_agent(obs, rng=None):
    """Completely random agent — establishes floor performance."""
    rng = rng or random
    score = round(rng.uniform(0.0, 1.0), 2)
    return Action(
        score=score,
        feedback="The prompt may have some issues with clarity and specificity.",
        improved_prompt=None,
    )


def heuristic_agent(obs):
    """
    Rule-based heuristic agent.
    - Short prompts → low scores
    - Checks for common quality markers (context, format, constraints)
    """
    prompt = obs.prompt
    words = prompt.split()
    word_count = len(words)

    # Base score from length
    if word_count <= 5:
        score = 0.10
    elif word_count <= 10:
        score = 0.25
    elif word_count <= 20:
        score = 0.45
    else:
        score = 0.60

    issues = []
    if word_count < 8:
        issues.append("too short and vague")
    if not any(k in prompt.lower() for k in ["format", "list", "step", "table", "bullet", "json"]):
        issues.append("no output format specified")
    if not any(k in prompt.lower() for k in ["context", "background", "i am", "my", "the following"]):
        issues.append("missing context or background")
    if not any(k in prompt.lower() for k in ["example", "e.g.", "for instance", "like"]):
        issues.append("no examples provided")
    if not any(k in prompt.lower() for k in ["tone", "style", "professional", "friendly", "concise"]):
        issues.append("no tone or style guidance")

    feedback = f"Prompt analysis: {', '.join(issues) if issues else 'Generally acceptable'}. "
    feedback += f"Word count: {word_count}. "
    if issues:
        feedback += f"Recommend adding: {', '.join(issues[:3])}."

    improved = None
    if word_count < 15:
        improved = (
            f"{prompt.strip()}. "
            "Please provide a detailed response with clear structure. "
            "Include relevant context, specify the desired format (e.g., bullet points or numbered steps), "
            "and note any constraints such as length, tone, or audience."
        )

    return Action(score=score, feedback=feedback, improved_prompt=improved)


def oracle_agent(obs):
    """
    Near-oracle agent — uses difficulty metadata to approximate ground truth.
    Simulates a well-calibrated expert reviewer.
    """
    difficulty_scores = {"easy": 0.12, "medium": 0.45, "hard": 0.40}
    base = difficulty_scores.get(obs.difficulty, 0.30)

    feedback_map = {
        "easy": (
            "This prompt is critically underdeveloped. It lacks: specific context, "
            "desired output format, length constraints, audience specification, and any "
            "examples. A good prompt should tell the AI exactly what to do, how to format "
            "the response, and provide all necessary background information."
        ),
        "medium": (
            "The prompt has a reasonable starting point but is missing important constraints. "
            "Key improvements needed: specify output format explicitly, add length guidelines, "
            "define tone and audience, and include at least one example of desired output. "
            "The task description is present but lacks the precision needed for reliable results."
        ),
        "hard": (
            "This complex prompt has a basic structure but is missing critical elements for "
            "production use. Issues include: no role definition, missing scope boundaries, "
            "no edge case handling, no few-shot examples, no output schema, and no "
            "chain-of-thought instructions. System prompts and meta-prompts require exhaustive "
            "specification to function reliably."
        ),
    }

    improved_templates = {
        "easy": f"[Improved version of: '{obs.prompt}']\n\nPlease provide a detailed, structured response to the following task: {obs.task_description}. Include: relevant context, step-by-step format, and specific constraints.",
        "medium": f"[Improved version]\nTask: {obs.task_description}\n\nOriginal prompt improved with: explicit format (numbered list), tone guidance (professional), length constraint (under 300 words), and audience specification (general reader).\n\n{obs.prompt} Please structure your response as a numbered list, keep it under 300 words, and use a professional tone.",
        "hard": f"[System Prompt - Improved]\nRole: You are an expert assistant specializing in: {obs.task_description}\n\nScope: [Defined]\nTone: Professional, clear, concise\nFormat: Structured with headers\nExamples: [Provided inline]\nEdge cases: Escalate ambiguous requests\nConstraints: Stay within defined topic area\n\nOriginal intent: {obs.prompt}",
    }

    return Action(
        score=base,
        feedback=feedback_map.get(obs.difficulty, "General feedback."),
        improved_prompt=improved_templates.get(obs.difficulty),
    )


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_agent(name, agent_fn, seed=42):
    """Run a single agent and return results."""
    rng = random.Random(seed)

    def seeded_agent(obs):
        if name == "random":
            return agent_fn(obs, rng)
        return agent_fn(obs)

    grader = PromptQualityGrader()
    results = grader.grade(seeded_agent)
    return results


def print_results(name, results):
    print(f"\n{'='*55}")
    print(f"  AGENT: {name.upper()}")
    print(f"{'='*55}")
    print(f"  Final Score : {results['total_score']:.4f}")
    print(f"  Passed      : {'✅ YES' if results['passed'] else '❌ NO'} (threshold: 0.50)")
    print(f"  Tasks Run   : {results['num_tasks']}")
    print()
    for i, (score, bd) in enumerate(zip(results["per_task_scores"], results["breakdown_per_task"])):
        diff = ["easy", "easy", "easy", "medium", "medium", "medium", "hard", "hard", "hard"][i]
        print(f"  Task {i+1:02d} [{diff:6s}] → {score:.3f} "
              f"| accuracy={bd.get('score_accuracy',0):.2f} "
              f"feedback={bd.get('feedback_quality',0):.2f} "
              f"improvement={bd.get('improvement_quality',0):.2f}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline agents on Prompt Quality Reviewer")
    parser.add_argument("--agent", choices=["random", "heuristic", "oracle", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agents = {
        "random": random_agent,
        "heuristic": heuristic_agent,
        "oracle": oracle_agent,
    }

    print("\n🔍 PROMPT QUALITY REVIEWER — BASELINE EVALUATION")
    print(f"   Seed: {args.seed}")

    if args.agent == "all":
        for name, fn in agents.items():
            results = run_agent(name, fn, seed=args.seed)
            print_results(name, results)
    else:
        fn = agents[args.agent]
        results = run_agent(args.agent, fn, seed=args.seed)
        print_results(args.agent, results)

    print("="*55)
    print("  Baseline Reference Scores (seed=42):")
    print("    random    → ~0.18")
    print("    heuristic → ~0.35")
    print("    oracle    → ~0.55")
    print("="*55)
