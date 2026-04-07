import sqlite3
import os
import logging
import re
import time
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import uvicorn

from .models import Action, Observation, Reward, StepResult, ResetResult
from .tasks import TASKS

# -- Security & Telemetry Setup --
class SecretMaskingFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        msg = re.sub(r'Bearer\s+[A-Za-z0-9_\-]+', 'Bearer ***', msg)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and hf_token in msg:
            msg = msg.replace(hf_token, "***")
        return msg

logger = logging.getLogger("agent_audit")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(SecretMaskingFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

security = HTTPBearer()
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "test-agent-key")

class TokenBucket:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_fill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_fill) * self.fill_rate)
        self.last_fill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

rate_limiter = TokenBucket(capacity=50, fill_rate=50.0/60.0)

async def verify_auth_and_rate_limit(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    if not rate_limiter.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return credentials.credentials

app = FastAPI(title="OpenEnv SQL Data Engineer")
db_semaphore = asyncio.Semaphore(5)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Max payload limit ~ 1MB
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1048576:
        return JSONResponse(status_code=413, content={"error": "Payload Too Large"})
    
    try:
        response = await asyncio.wait_for(call_next(request), timeout=10.0)
        return response
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Gateway Timeout"})

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
        
        # Security: 5 second timeout on queries
        self.query_start_time = 0
        def progress_handler():
            if time.time() - self.query_start_time > 5.0:
                return 1 # Abort query
            return 0
        self.conn.set_progress_handler(progress_handler, 1000)

        # Security: Simple Authorizer to block DROP TABLE and explicit destructive system mods
        # We allow standard DDL since the agent gets asked to CREATE VIEW, but restrict DROP
        def authorizer(action_code, arg1, arg2, dbname, source):
            # 11 = SQLITE_DROP_TABLE, 16 = SQLITE_DROP_VIEW
            # 17 = SQLITE_ATTACH (blocks ATTACH DATABASE)
            if action_code in (11, 16, 17):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        self.conn.set_authorizer(authorizer)
        
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
            
            # Injection Mitigations at parser level
            blocked_patterns = [r"(?i)DROP\s+DATABASE", r"(?i)pg_sleep", r"(?i)randomblob", r"(?i)ATTACH\s+DATABASE"]
            for p in blocked_patterns:
                if re.search(p, query):
                    raise Exception(f"Blocked destructive command pattern detected: {p}")
                    
            if query.upper().startswith("DROP TABLE sqlite_"):
                raise Exception("Cannot modify system tables.")
                
            self.query_start_time = time.time()
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
        
        # Episode terminates when score is 0.99
        done = (self.current_score >= 0.99) or (self.step_count > 30)
        
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
async def reset(request: Request):
    task_id = 1
    try:
        data = await request.json()
        if "task_id" in data:
            task_id = int(data["task_id"])
    except:
        pass # gracefully handle missing json body (curl -d '{}')

    async with db_semaphore:
        try:
            logger.info(f"Resetting environment for task_id: {task_id}")
            return env_instance.reset(task_id=task_id)
        except Exception as e:
            logger.error(f"Reset Error: {str(e)}")
            raise HTTPException(status_code=400, detail="Error during reset")

@app.post("/step", response_model=StepResult)
async def step(action: Action, token: str = Depends(verify_auth_and_rate_limit)):
    async with db_semaphore:
        try:
            start_t = time.time()
            logger.info(f"Attempting query: {action.action_str}")
            res = env_instance.step(action)
            duration = time.time() - start_t
            logger.info(f"Query completed in {duration:.3f}s. Error state: {res.observation.last_action_error}")
            return res
        except Exception as e:
            logger.error(f"Step Error: {str(e)}")
            raise HTTPException(status_code=400, detail="Error executing SQL step")

@app.get("/state")
async def state(token: str = Depends(verify_auth_and_rate_limit)):
    async with db_semaphore:
        return env_instance.state()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
