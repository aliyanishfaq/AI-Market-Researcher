import copy

class SchemaTransformer:
    """
    A class to transform legacy JSON schemas into the format required by the new API.
    
    This class provides methods to:
      - Recursively add "additionalProperties": false to every object in the schema.
      - Update "$ref" values (e.g., converting "/definitions/" to "/$defs/").
      - Enforce that every object with "properties" has a "required" array listing every property.
      - Wrap a legacy schema into a new top-level structure.
    """

    @staticmethod
    def add_additional_properties(schema):
        """
        Recursively add "additionalProperties": false to every object in the schema.
        
        Parameters:
            schema (dict or list): The JSON schema (or part of it) to update.
        
        Returns:
            The updated schema with additionalProperties set to false for objects.
        """
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                # Only set additionalProperties if it isn’t already defined
                schema.setdefault("additionalProperties", False)
            for key, value in schema.items():
                if isinstance(value, dict):
                    SchemaTransformer.add_additional_properties(value)
                elif isinstance(value, list):
                    for item in value:
                        SchemaTransformer.add_additional_properties(item)
        elif isinstance(schema, list):
            for item in schema:
                SchemaTransformer.add_additional_properties(item)
        return schema

    @staticmethod
    def update_refs_to_defs(schema):
        """
        Recursively update "$ref" values to point to "$defs" instead of "definitions",
        if required by the new API.
        
        Parameters:
            schema (dict or list): The JSON schema (or part of it) to update.
        
        Returns:
            The updated schema with corrected $ref paths.
        """
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key == "$ref" and isinstance(value, str):
                    # Replace "/definitions/" with "/$defs/" if found.
                    schema[key] = value.replace("/definitions/", "/$defs/")
                else:
                    SchemaTransformer.update_refs_to_defs(value)
        elif isinstance(schema, list):
            for item in schema:
                SchemaTransformer.update_refs_to_defs(item)
        return schema

    @staticmethod
    def enforce_required_fields(schema):
        """
        Recursively enforce that every object with "properties" includes a "required" array
        listing every property key. This is required by the new API.
        
        Parameters:
            schema (dict or list): The JSON schema (or part of it) to update.
        
        Returns:
            The updated schema with "required" arrays for objects.
        """
        if isinstance(schema, dict):
            if schema.get("type") == "object" and "properties" in schema:
                # Overwrite (or add) "required" to include every property key.
                schema["required"] = list(schema["properties"].keys())
            for key, value in schema.items():
                if isinstance(value, (dict, list)):
                    SchemaTransformer.enforce_required_fields(value)
        elif isinstance(schema, list):
            for item in schema:
                SchemaTransformer.enforce_required_fields(item)
        return schema

    def wrap_schema(self, original_schema, name, description, strict=True):
        """
        Wrap the legacy schema into the new API's expected format.
        
        The new schema format will include keys like "name", "description", and "strict",
        and the original schema will be nested under the "schema" key.
        
        Parameters:
            original_schema (dict): The legacy JSON schema to transform.
            name (str): The name to assign to the wrapped schema.
            description (str): A description for the wrapped schema.
            strict (bool): Whether to enforce strict schema validation (default True).
        
        Returns:
            A new dictionary representing the wrapped schema.
        """
        # Work on a copy so as not to modify the original schema.
        schema_copy = copy.deepcopy(original_schema)
        
        # Recursively add additionalProperties: false to every object in the schema.
        updated_schema = self.add_additional_properties(schema_copy)
        
        # Update any references from "definitions" to "$defs" if needed.
        updated_schema = self.update_refs_to_defs(updated_schema)
        
        # Enforce that every object with "properties" has a "required" array
        updated_schema = self.enforce_required_fields(updated_schema)
        
        # Wrap the updated schema in the new format.
        wrapped_schema = {
            "name": name,
            "description": description,
            "strict": strict,
            "schema": updated_schema
        }
        return wrapped_schema

    def wrap_as_step(self, explanation_schema, output_schema):
        """
        Wrap two schemas into a 'step' format as required by one of the new API examples.
        
        This method creates a $defs block for a step and then references it.
        
        Parameters:
            explanation_schema (dict): The schema for the explanation.
            output_schema (dict): The schema for the output.
        
        Returns:
            A dictionary representing the step-wrapped schema.
        """
        step_schema = {
            "$defs": {
                "step": {
                    "type": "object",
                    "properties": {
                        "explanation": explanation_schema,
                        "output": output_schema
                    },
                    "required": ["explanation", "output"],
                    "additionalProperties": False
                }
            },
            "$ref": "#/$defs/step"
        }
        return step_schema

# Example usage:
if __name__ == "__main__":
    # Define a legacy schema (example: THEME_RADAR_SCHEMA with theme_connections missing "required")
    THEME_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "strength": {"type": "number"},
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "mixed"]
            },
            "frequency": {"type": "integer"},
            "supporting_quotes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "related_themes": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "strength", "sentiment", "frequency", "supporting_quotes", "related_themes"]
    }

    THEME_RADAR_SCHEMA = {
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": THEME_SCHEMA
            },
            "primary_theme": {"type": "string"},
            "theme_connections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"}
                    }
                    # Note: no "required" was originally provided here.
                }
            }
        },
        "required": ["themes", "primary_theme", "theme_connections"]
    }

    transformer = SchemaTransformer()
    new_theme_radar_schema = transformer.wrap_schema(
        original_schema=THEME_RADAR_SCHEMA,
        name="theme",
        description="Schema for Gemini qualitative analytics theme radar"
    )

    import json
    print(json.dumps(new_theme_radar_schema, indent=2))
