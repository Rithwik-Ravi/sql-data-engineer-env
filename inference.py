import os
import json
from openai import OpenAI
from server.models import Action, BrowserGymAction # using our local Action model alias
from server.app import env_instance as env

# Environment Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

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
    text = response_text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    
    try:
        data = json.loads(text)
        return data.get("action_str", "SELECT 1;")
    except json.JSONDecodeError:
        return text

def run_task(task_id: int):    
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )
    
    history = []
    rewards = []
    
    try:
        result = env.reset(task_id=task_id)
        observation = result.observation
        final_score = result.info.get("initial_score", 0.0)
    except Exception as e:
        print(f"[START] task={task_id} env=sql-data-engineer-env model={MODEL_NAME}")
        print(f"[END] success=false steps=0 score=0.00 rewards=")
        return 0.0

    print(f"[START] task={task_id} env=sql-data-engineer-env model={MODEL_NAME}")

    done = False
    step_count = 0

    for step in range(1, MAX_STEPS + 1):
        step_count = step
        user_prompt = build_user_prompt(step, observation, history)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        action_str = ""
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                response_format={"type": "json_object"} 
            )
            response_text = completion.choices[0].message.content or ""
            action_str = parse_model_action(response_text)
        except Exception as exc: 
            action_str = "SELECT 1;"

        try:
            step_result = env.step(BrowserGymAction(action_str=action_str))
            observation = step_result.observation
            reward = step_result.reward
            done = step_result.done
            final_score = step_result.info.get("current_score", 0.0)
            
            if observation.last_action_error:
                error_msg = observation.result.replace('\n', ' ')
            else:
                error_msg = "null"
        except Exception as e:
            reward = 0.0
            done = True
            error_msg = str(e).replace('\n', ' ')

        rewards.append(f"{reward:.2f}")
        
        done_str = "true" if done else "false"
        safe_action = action_str.replace('\n', ' ')
        err_out = f'"{error_msg}"' if error_msg != "null" else "null"
        
        print(f"[STEP] step={step} action=\"{safe_action}\" reward={reward:.2f} done={done_str} error={err_out}")

        history_line = f"Step {step}: {safe_action[:50]}... -> reward {reward:+.2f}"
        history.append(history_line)

        if done:
            break

    success_str = "true" if final_score >= 1.0 else "false"
    rewards_str = ",".join(rewards)
    print(f"[END] success={success_str} steps={step_count} score={final_score:.2f} rewards={rewards_str}")
    
    return final_score

def main():    
    for task_id in [1, 2, 3]:
        run_task(task_id)

if __name__ == "__main__":
    main()
