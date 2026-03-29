import requests
from typing import Dict, Any, Optional

from server.models import Action, StepResult, ResetResult

class EnvironmentalClient:
    """
    Official OpenEnv client wrapper to interface with the containerized environment endpoints.
    Can be used by baseline inference scripts or remote agent evaluations.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url.rstrip('/')
        
    def reset(self, task_id: int = 1) -> ResetResult:
        """
        Calls the /reset endpoint to begin a new episode for the specified task.
        """
        response = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        response.raise_for_status()
        return ResetResult(**response.json())
        
    def step(self, action: Action) -> StepResult:
        """
        Submits a step (action) to the environment and returns the updated state/reward.
        """
        # Pydantic dict serialization for requests
        response = requests.post(f"{self.base_url}/step", json=action.dict())
        response.raise_for_status()
        return StepResult(**response.json())
        
    def state(self) -> Dict[str, Any]:
        """
        Extracts the unstructured metadata state of the running environment.
        """
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()

# Provide a default singleton for ease of use
client = EnvironmentalClient()
