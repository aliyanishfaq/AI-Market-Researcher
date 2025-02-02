from typing import List, Dict
import json
from pydantic import BaseModel
from prompts import build_employee_prompt_v1, build_employee_prompt_v2, build_employee_prompt_v3, build_employee_prompt_v4, build_employee_personality_summary_prompt
from schema import Persona
import random

class PersonaManager:
    def __init__(self, data_source: str = "glassdoor"):
        self._personas: Dict[str, Persona] = {}
        self._conversation_memory: Dict[str, List[Dict[str, str]]] = {}
        self.employee_prompt_variations = [build_employee_prompt_v1, build_employee_prompt_v2, build_employee_prompt_v3, build_employee_prompt_v4]
        self.data_source = data_source
        self._load_personas()

    def _load_personas(self):
        """Load personas from JSON file"""
        try:
            with open(f"{self.data_source}.json", "r") as f:
                personas_data = json.load(f)
                for idx, data in enumerate(personas_data):
                    persona_id = str(idx)
                    self._personas[persona_id] = Persona(
                        id=persona_id,
                        conversation_history=[],
                        **data
                    )
        except FileNotFoundError:
            raise Exception("Personas data file not found")
        except Exception as e:
            print(f"Error loading persona: {e}")  # Debug print
            raise

    def get_all_personas(self) -> List[Persona]:
        """Return all personas"""
        return list(self._personas.values())

    def get_persona(self, persona_id: str) -> Persona:
        """Get a specific persona"""
        return self._personas[persona_id]

    def update_conversation_history(self, persona_id: str, question: str, distribution: Dict[str, float]):
        """Update persona's conversation history"""
        persona = self._personas[persona_id]
        summary = self._create_response_summary(question, distribution)
        persona.conversation_history.append({
            "question": question,
            "summary": summary
        })

    def update_personality_summary(self, persona_id: str, personality_summary: str):
        """Update persona's personality summary"""
        persona = self._personas[persona_id]
        persona.personality_summary = personality_summary

    def _create_response_summary(self, question: str, distribution: Dict[str, float]) -> str:
        """Create a summary of the response for context memory"""
        max_option = max(distribution.items(), key=lambda x: x[1])
        return f"When asked '{question}', leaned {int(max_option[1]*100)}% towards '{max_option[0]}'"
        
    def _build_employee_prompt(self, persona: Persona, question: str, options: List[str]) -> str:
        """Build a prompt of employee for the LLM including persona context and conversation history"""
        selected_prompt_builder = random.choice(self.employee_prompt_variations)
        return selected_prompt_builder(persona, question, options)


    def build_prompt(self, persona_id: str, question: str, options: List[str]) -> str:
        """Build a prompt for the LLM including persona context and conversation history"""
        persona = self._personas[persona_id]
        if self.data_source == "glassdoor":
            prompt = self._build_employee_prompt(persona, question, options)
        else:
            prompt = self._build_product_prompt(persona, question, options)

        return prompt

    def get_personality_summary_prompt(self, persona_id: str) -> str:
        """Get a summary of the personality of a persona"""
        persona = self._personas[persona_id]
        prompt = build_employee_personality_summary_prompt(persona)
        
        return prompt