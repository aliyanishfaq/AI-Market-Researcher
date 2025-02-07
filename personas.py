from typing import List, Dict
import json
from pydantic import BaseModel
from prompts import build_employee_prompt_v1, build_employee_prompt_v2, build_employee_prompt_v3, build_employee_prompt_v4, build_employee_personality_summary_prompt, build_product_reviewer_prompt_v1, build_product_reviewer_prompt_v2, build_product_reviewer_prompt_v3, build_product_reviewer_prompt_v4, build_product_reviewer_personality_summary_prompt
from schema import Persona, PersonaType
import random

class PersonaManager:
    def __init__(self, persona_type: PersonaType):
        self._personas: Dict[str, Persona] = {}
        self._conversation_memory: Dict[str, List[Dict[str, str]]] = {}
        self.persona_type = persona_type
        self.employee_prompt_variations = [build_employee_prompt_v1, build_employee_prompt_v2, 
                                        build_employee_prompt_v3, build_employee_prompt_v4]
        self.product_reviewer_prompt_variations = [build_product_reviewer_prompt_v1, 
                                                 build_product_reviewer_prompt_v2, 
                                                 build_product_reviewer_prompt_v3, 
                                                 build_product_reviewer_prompt_v4]

        if persona_type == PersonaType.INTEL_EMPLOYEE:
            self.data_source = "glassdoor.json"
        else:
            self.data_source = "product-reviews.json"
        self._load_personas()

    def _load_personas(self):
        """Load personas from JSON file"""
        try:
            with open(f"{self.data_source}", "r") as f:
                personas_data = json.load(f)
                for idx, data in enumerate(personas_data):
                    persona_id = str(idx)
                    
                    if self.persona_type == PersonaType.INTEL_EMPLOYEE:
                        persona_data = {
                            "id": persona_id,
                            "name": data.get("name"),
                            "rating": data.get("rating"),
                            "date": data.get("date"),
                            "title": data.get("title"),
                            "role": data.get("role"),
                            "location": data.get("location"),
                            "recommend": data.get("recommend"),
                            "employment_status": data.get("employment_status"),
                            "pros": data.get("pros"),
                            "cons": data.get("cons"),
                            "advice_to_management": data.get("advice_to_management"),
                            "conversation_history": []
                        }
                    else:  # INTEL_PRODUCT_REVIEWER
                        review_data = data.get("review", {})
                        product_data = data.get("product", {})
                        user_context = data.get("user_context", {})
                        
                        persona_data = {
                            "id": persona_id,
                            "name": review_data.get("name"),
                            "rating": review_data.get("rating", {}).get("score"),
                            "date": review_data.get("publication_date"),
                            "title": review_data.get("title"),
                            "summary": review_data.get("summary"),
                            "pros": review_data.get("pros", []),
                            "cons": review_data.get("cons", []),
                            "recommend": review_data.get("recommend"),
                            "themes": review_data.get("themes", []),
                            "suggestions": review_data.get("suggestions", []),
                            "product_name": product_data.get("name"),
                            "product_category": product_data.get("category"),
                            "manufacturer": product_data.get("manufacturer"),
                            "location": product_data.get("location"),
                            "use_case": user_context.get("use_case"),
                            "technical_level": user_context.get("technical_level"),
                            "conversation_history": []
                        }

                    self._personas[persona_id] = Persona(**persona_data)

        except FileNotFoundError:
            raise Exception(f"Personas data file {self.data_source} not found")
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error loading persona: {e}")  # Debug print
            print(f"Error traceback: {error_trace}")
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

    def _build_product_reviewer_prompt(self, persona: Persona, question: str, options: List[str]) -> str:
        """Build a prompt of product reviewer for the LLM including persona context and conversation history"""
        selected_prompt_builder = random.choice(self.product_reviewer_prompt_variations)
        return selected_prompt_builder(persona, question, options)

    def build_prompt(self, persona_id: str, question: str, options: List[str]) -> str:
        """Build a prompt for the LLM including persona context and conversation history"""
        persona = self._personas[persona_id]
        if self.persona_type == PersonaType.INTEL_EMPLOYEE:
            prompt = self._build_employee_prompt(persona, question, options)
        elif self.persona_type == PersonaType.INTEL_PRODUCT_REVIEWER:
            prompt = self._build_product_reviewer_prompt(persona, question, options)
        else:
            raise ValueError(f"Invalid persona type: {self.persona_type}")

        return prompt

    def get_personality_summary_prompt(self, persona_id: str) -> str:
        """Get a summary of the personality of a persona"""
        persona = self._personas[persona_id]
        if self.persona_type == PersonaType.INTEL_EMPLOYEE:
            prompt = build_employee_personality_summary_prompt(persona)
        elif self.persona_type == PersonaType.INTEL_PRODUCT_REVIEWER:
            prompt = build_product_reviewer_personality_summary_prompt(persona)
        else:
            raise ValueError(f"Invalid persona type: {self.persona_type}")
        
        return prompt