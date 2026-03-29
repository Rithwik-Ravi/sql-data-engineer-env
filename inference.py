import os
import json
from openai import OpenAI
from server.models import Action, BrowserGymAction # using our local Action model alias
from server.app import env_instance as env

# Environment Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")

MAX_STEPS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 512

SYSTEM_PROMPT = """
You are an expert Data Engineer interacting with a simulated SQLite database.
You will be given a task goal, the current database schema, and the most recent step's SQL output or error.
Your goal is to complete the task by executing SQL commands. 

CRITICAL RULES:
1. You may only execute ONE SQL statement at a time. Do not chain statements with semicolons.
2. If you need to review data, use short SELECT queries.
3. If your previous action resulted in an SQL error, fix the error and try again.
4. If you need multiple steps to achieve the goal (e.g. create tables, then insert data), execute them one by one.
5. You MUST output ONLY a valid JSON object matching this schema:
{
  "action_str": "YOUR SQL QUERY HERE"
}
Do not wrap your response in markdown code blocks. Just valid JSON.
"""

def build_user_prompt(step: int, observation, history: list) -> str:
    prompt = f"--- Step {step} ---\n"
    prompt += f"Goal: {observation.goal}\n\n"
    if observation.schema_dump:
        prompt += f"Current DB Schema:\n{observation.schema_dump}\n\n"
    
    prompt += f"Last Result (or Error):\n{observation.result}\n\n"
    
    if history:
        prompt += "Action History (Last 3 steps):\n"
        for h in history[-3:]:
            prompt += h + "\n"
            
    prompt += "\nProvide the JSON with your next `action_str`:"
    return prompt

def parse_model_action(response_text: str) -> str:
    # Try to parse JSON
    text = response_text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    
    try:
        data = json.loads(text)
        return data.get("action_str", "SELECT 1;")
    except json.JSONDecodeError:
        # Fallback if model doesn't follow json format correctly
        return text

def run_task(task_id: int):
    print(f"\n{'='*50}\nStarting Task {task_id}\n{'='*50}")
    
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
    
    history = []
    
    # Using the local env object wrapper
    result = env.reset(task_id=task_id)
    observation = result.observation
    print(f"Episode goal: {observation.goal}\n")

    for step in range(1, MAX_STEPS + 1):
        # We handle done from the step result, but for initial step we check just in case
        user_prompt = build_user_prompt(step, observation, history)
        
        # print("PROMPT:", user_prompt)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                response_format={"type": "json_object"} # enforce json output
            )
            response_text = completion.choices[0].message.content or ""
            action_str = parse_model_action(response_text)
        except Exception as exc: 
            failure_msg = f"Model request failed ({exc}). Using fallback action."
            print(failure_msg)
            action_str = "SELECT 1;"

        print(f"Step {step}: model suggested -> {action_str[:100]}...")

        # Step the environment
        step_result = env.step(BrowserGymAction(action_str=action_str))
        observation = step_result.observation
        reward = step_result.reward
        done = step_result.done

        error_flag = " ERROR" if observation.last_action_error else ""
        history_line = f"Step {step}: {action_str[:50]}... -> reward {reward:+.2f}{error_flag}"
        history.append(history_line)
        
        print(f"  Reward: {reward:+.2f} | Done: {done} | Last action error: {observation.last_action_error}")

        if done:
            final_score = step_result.info.get("current_score", 0.0)
            print(f"\nEpisode complete! Final Score: {final_score}/1.0")
            break
    else:
        final_score = env.state().get("current_score", 0.0)
        print(f"\nReached max steps ({MAX_STEPS}). Final Score: {final_score}/1.0")
        
    return final_score

def main():
    print("Testing OpenEnv Data Engineer Inference Baseline")
    
    if not API_KEY:
        print("Warning: API_KEY/HF_TOKEN not set. Will likely fail unless local LLM.")

    scores = {}
    for task_id in [1, 2, 3]:
        score = run_task(task_id)
        scores[f"Task_{task_id}"] = score
        
    print(f"\n{'*'*50}\nEVALUATION COMPLETE\n{scores}\n{'*'*50}")

if __name__ == "__main__":
    main()
