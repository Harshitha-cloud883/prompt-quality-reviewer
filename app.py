"""
Gradio App — Prompt Quality Reviewer (OpenEnv)
Hugging Face Spaces entry point.
"""

import gradio as gr
from environment import PromptQualityEnv, Action

env = PromptQualityEnv()
current_obs = None
session_log = []


def start_env():
    global current_obs, session_log
    session_log = []
    current_obs = env.reset()
    return _format_obs(current_obs), "", "", "", _format_log()


def submit_action(score_str, feedback, improved_prompt):
    global current_obs, session_log

    if current_obs is None:
        return "Please start the environment first.", "", "", "", ""

    try:
        score = float(score_str)
    except ValueError:
        return "Score must be a number between 0.0 and 1.0", "", "", "", _format_log()

    action = Action(
        score=score,
        feedback=feedback,
        improved_prompt=improved_prompt if improved_prompt.strip() else None,
    )

    result = env.step(action)
    current_obs = result.observation

    bd = result.info.get("reward_breakdown", {})
    log_entry = (
        f"Task reward: {result.reward:.3f} | "
        f"Score acc: {bd.get('score_accuracy',0):.2f} | "
        f"Feedback: {bd.get('feedback_quality',0):.2f} | "
        f"Improvement: {bd.get('improvement_quality',0):.2f}"
    )
    session_log.append(log_entry)

    if result.done:
        state = env.state()
        obs_text = f"✅ All tasks complete!\nTotal reward: {state.total_reward:.3f}\nTasks completed: {state.tasks_completed}"
    else:
        obs_text = _format_obs(current_obs)

    return obs_text, "", "", "", _format_log()


def _format_obs(obs):
    return (
        f"📋 TASK [{obs.difficulty.upper()}] — Step {obs.step_number}/{obs.max_steps}\n\n"
        f"Goal: {obs.task_description}\n\n"
        f"Prompt to evaluate:\n\"{obs.prompt}\"\n\n"
        + (f"Your previous feedback: {obs.previous_feedback}" if obs.previous_feedback else "")
    )


def _format_log():
    if not session_log:
        return "No steps yet."
    return "\n".join(f"{i+1}. {entry}" for i, entry in enumerate(session_log))


with gr.Blocks(title="Prompt Quality Reviewer — OpenEnv") as demo:
    gr.Markdown("# 🔍 Prompt Quality Reviewer\n**OpenEnv × Meta × Scaler Hackathon**\n\nEvaluate and improve AI prompts across 9 tasks (easy → medium → hard).")

    with gr.Row():
        with gr.Column(scale=2):
            obs_box = gr.Textbox(label="Current Task", lines=8, interactive=False)
            score_input = gr.Textbox(label="Your Score (0.0 – 1.0)", placeholder="e.g. 0.25")
            feedback_input = gr.Textbox(label="Feedback (identify issues)", lines=4, placeholder="The prompt is too vague because...")
            improved_input = gr.Textbox(label="Improved Prompt (optional)", lines=4, placeholder="Rewrite the prompt here...")
            with gr.Row():
                start_btn = gr.Button("▶ Start / Reset", variant="secondary")
                submit_btn = gr.Button("Submit Action →", variant="primary")
        with gr.Column(scale=1):
            log_box = gr.Textbox(label="Session Log", lines=20, interactive=False)

    start_btn.click(start_env, outputs=[obs_box, score_input, feedback_input, improved_input, log_box])
    submit_btn.click(submit_action, inputs=[score_input, feedback_input, improved_input],
                     outputs=[obs_box, score_input, feedback_input, improved_input, log_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
