from pydantic import BaseModel
from typing import Optional, Any, Dict, List

class Action(BaseModel):
    action_str: str
    
# Alias for compatibility with the sample inference script
BrowserGymAction = Action

class Observation(BaseModel):
    goal: str
    result: str
    step: int
    last_action_error: bool
    schema_dump: Optional[str] = None
    
class Reward(BaseModel):
    value: float
    reason: Optional[str] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResult(BaseModel):
    observation: Observation
    info: Dict[str, Any]
