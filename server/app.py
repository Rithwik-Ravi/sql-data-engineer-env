import sqlite3
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .models import Action, Observation, Reward, StepResult, ResetResult
from .tasks import TASKS

app = FastAPI(title="OpenEnv SQL Data Engineer")

class SQLEnvironment:
    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None
        self.task_id = 1
        self.step_count = 0
        self.current_score = 0.0
        self.db_path = "tmp_env.db"

    def get_schema_dump(self) -> str:
        if not self.conn:
            return ""
        try:
            c = self.conn.cursor()
            c.execute("SELECT type, name, sql FROM sqlite_master WHERE type='table' OR type='view'")
            rows = c.fetchall()
            dump = []
            for t, name, sql in rows:
                if name.startswith('sqlite_'): continue
                dump.append(f"[{t.upper()}] {name}:\n  {sql}")
                
            return "\n".join(dump) if dump else "Database is empty."
        except Exception as e:
            return f"Error extracting schema: {e}"

    def reset(self, task_id: int = 1) -> ResetResult:
        if self.conn:
            self.conn.close()
        
        # Clean up existing temp db
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.task_id = task_id
        if self.task_id not in TASKS:
            raise ValueError(f"Task ID {self.task_id} not found.")
            
        self.conn = sqlite3.connect(self.db_path)
        self.step_count = 0
        self.current_score = 0.0
        
        # Initialize standard SQLite settings
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Setup specific task data
        task = TASKS[self.task_id]
        task.setup_db(self.conn)
        self.current_score = task.grade(self.conn) # Base score
        
        goal_text = task.get_goal()
        # Add basic info about actual task goal
        instructions = f"Task Goal: {goal_text}\n"
        
        obs = Observation(
            goal=instructions,
            result="Environment initialized. Schema ready.",
            step=self.step_count,
            last_action_error=False,
            schema_dump=self.get_schema_dump()
        )
        return ResetResult(observation=obs, info={"task_id": self.task_id, "initial_score": self.current_score})

    def step(self, action: Action) -> StepResult:
        if not self.conn:
            raise ValueError("Environment not initialized. Call reset() first.")
            
        self.step_count += 1
        last_action_error = False
        query_result = ""
        
        try:
            c = self.conn.cursor()
            query = action.action_str.strip()
            # Basic mitigation of forbidden queries just in case (though we're in mock)
            if query.upper().startswith("DROP TABLE sqlite_"):
                raise Exception("Cannot modify system tables.")
                
            c.execute(query)
            
            if query.upper().startswith("SELECT") or query.upper().startswith("PRAGMA"):
                rows = c.fetchmany(10) # limit output size for LLM observation
                col_names = [description[0] for description in c.description] if c.description else []
                # Format tabular output
                result_str = " | ".join(col_names) + "\n"
                result_str += "-" * len(result_str) + "\n"
                for r in rows:
                    result_str += " | ".join([str(val) for val in r]) + "\n"
                if len(rows) == 10:
                    result_str += "... (output truncated)"
                query_result = result_str
            else:
                self.conn.commit()
                query_result = f"Command executed successfully. Rowcount: {c.rowcount}"
                
        except Exception as e:
            last_action_error = True
            query_result = f"SQL Error: {str(e)}"
            
        # Run grader
        task = TASKS[self.task_id]
        new_score = task.grade(self.conn)
        
        # Reward is dense: change in score + small penalty for errors
        reward_value = new_score - self.current_score
        
        if last_action_error:
            # minor penalty for syntax errors
            reward_value -= 0.05
            
        self.current_score = new_score
        
        # Episode terminates when score is 1.0
        done = (self.current_score >= 1.0) or (self.step_count > 30)
        
        obs = Observation(
            goal=task.get_goal(),
            result=query_result,
            step=self.step_count,
            last_action_error=last_action_error,
            schema_dump=self.get_schema_dump() if not last_action_error else None # only dump if no error to save tokens
        )
        
        return StepResult(
            observation=obs,
            reward=reward_value,
            done=done,
            info={"current_score": self.current_score}
        )

    def state(self) -> Any:
        # Return state as unstructured dict per standard API
        return {
            "task_id": self.task_id,
            "step": self.step_count,
            "current_score": self.current_score,
            "schema_dump": self.get_schema_dump()
        }

# Global instance
env_instance = SQLEnvironment()

class ResetRequest(BaseModel):
    task_id: int = 1

@app.post("/reset", response_model=ResetResult)
def reset(req: ResetRequest):
    try:
        return env_instance.reset(task_id=req.task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        return env_instance.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return env_instance.state()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
