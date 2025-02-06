import json
from typing import List, Dict
from pathlib import Path

class PersonaLoader:
    _instance = None
    _personas = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PersonaLoader, cls).__new__(cls)
            cls._instance._load_all_personas()
        return cls._instance

    def _load_all_personas(self):
        # Get the project root directory
        root_dir = Path(__file__).parent.parent

        persona_files = {
            'intel_employee': root_dir / 'glassdoor.json',
            'intel_product_reviewer': root_dir / 'product-reviews.json',
        }

        # Load all persona files
        for persona_type, file_path in persona_files.items():
            try:
                with open(file_path, 'r') as f:
                    self._personas[persona_type] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Persona file {file_path} not found")
                self._personas[persona_type] = []

    def get_personas(self, persona_type: str) -> List[dict]:
        return self._personas.get(persona_type, [])

    def get_persona(self, persona_type: str, index: int) -> Dict:
        personas = self.get_personas(persona_type)
        if not personas or index >= len(personas):
            raise ValueError(f"Persona not found for type {persona_type} at index {index}")
        return personas[index]