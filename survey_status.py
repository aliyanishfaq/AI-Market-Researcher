from enum import Enum
from datetime import datetime
from typing import List
from pydantic import BaseModel

class SurveyStage(Enum):
    INITIALIZING = "Initializing survey"
    BUILDING_PROMPTS = "Fetching Personas"
    QUERYING_LLM = "Surveying Personas"
    QUANTITATIVE_ANALYSIS = "Sampling Responses"
    QUALITATIVE_ANALYSIS = "Analyzing Responses"
    COMPLETED = "Completed"
    ERROR = "Error"

class SimulationStatus(BaseModel):
    """Status tracking for the simulation"""
    start_time: datetime
    current_question: int
    total_questions: int
    completed_personas: int
    total_personas: int
    stage: SurveyStage = SurveyStage.INITIALIZING
    message: str = ""
    errors: List[str] = []

    def update(self, stage: SurveyStage, message: str = "", error: str = None):
        self.stage = stage
        self.message = message
        if error:
            self.errors.append(error)
            self.stage = SurveyStage.ERROR
        print(f"[Survey Status] {stage.value}: {message}") 