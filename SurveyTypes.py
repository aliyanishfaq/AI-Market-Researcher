from pydantic import BaseModel
from typing import List, Dict
from schema import PersonaType


class Option(BaseModel):
    id: str
    text: str

class Question(BaseModel):
    id: str
    text: str
    options: List[Option]

class SurveyRequest(BaseModel):
    title: str
    questions: List[Question]
    persona_type: PersonaType
    number_of_personas: int = 5
    number_of_samples: int = 2000