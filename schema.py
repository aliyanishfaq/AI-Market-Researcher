from typing import List, Dict, Any, Literal, Union
from typing_extensions import TypedDict
from pydantic import BaseModel
import enum

class Persona(BaseModel):
    id: str
    name: str = "Anonymous"  # Default for product reviews
    date: str
    title: str
    rating: Union[float, None] = None  # Use Union for type unions
    recommend: Union[bool, None] = None
    
    # Fields specific to employee reviews
    role: Union[str, None] = None
    location: Union[str, None] = None
    employment_status: Union[str, None] = None
    ceo_approval: Union[bool, None] = None
    business_outlook: Union[bool, None] = None
    
    # Fields specific to product reviews
    pros: Union[List[str], str]  # Use Union for type unions
    cons: Union[List[str], str]  # Use Union for type unions
    themes: Union[List[str], None] = None
    suggestions: Union[List[str], None] = None
    
    # Product-specific fields
    product: Union[Dict[str, Any], None] = None  # Use Union for type unions
    user_context: Union[Dict[str, Any], None] = None  # Use Union for type unions
    publication_date: Union[str, None] = None  # Use Union for type unions
    
    # Common fields that might have different names
    advice_to_management: Union[str, None] = None  # Use Union for type unions
    summary: Union[str, None] = None  # Use Union for type unions
    
    # Conversation tracking
    conversation_history: List[Dict[str, str]] = []
    personality_summary: Union[str, None] = None
    @property
    def persona_type(self) -> str:
        """Determine if the object is an employee or product review"""
        return "employee" if self.role is not None else "product"



# Schemas for Gemini qualitative analytics
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
            }
        }
    },
    "required": ["themes", "primary_theme", "theme_connections"]
}

PERSONA_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "role": {"type": "string"},
        "experience_level": {
            "type": "string",
            "enum": ["junior", "mid", "senior", "leadership"]
        },
        "sentiment_score": {"type": "number"},
        "key_concerns": {
            "type": "array",
            "items": {"type": "string"}
        },
        "primary_response": {"type": "string"}
    },
    "required": ["id", "role", "experience_level", "sentiment_score", "key_concerns", "primary_response"]
}

PERSONA_CONNECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "strength": {"type": "number"},
        "shared_views": {
            "type": "array",
            "items": {"type": "string"}
        },
        "divergent_views": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["source_id", "target_id", "strength", "shared_views", "divergent_views"]
}

PERSONA_NETWORK_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": PERSONA_NODE_SCHEMA
        },
        "connections": {
            "type": "array",
            "items": PERSONA_CONNECTION_SCHEMA
        },
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "members": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "members"]
            }
        }
    },
    "required": ["nodes", "connections", "clusters"]
}

SENTIMENT_STAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "stage_name": {"type": "string"},
        "positive_score": {"type": "number"},
        "neutral_score": {"type": "number"},
        "negative_score": {"type": "number"},
        "key_drivers": {
            "type": "array",
            "items": {"type": "string"}
        },
        "common_phrases": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["stage_name", "positive_score", "neutral_score", "negative_score", "key_drivers", "common_phrases"]
}

SENTIMENT_FLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "stages": {
            "type": "array",
            "items": SENTIMENT_STAGE_SCHEMA
        },
        "trend": {
            "type": "string",
            "enum": ["improving", "declining", "stable", "mixed"]
        },
        "critical_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["stage", "description"]
            }
        }
    },
    "required": ["stages", "trend", "critical_points"]
}

HEATMAP_CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "string"},
        "y": {"type": "string"},
        "value": {"type": "number"},
        "count": {"type": "integer"},
        "notable_responses": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["x", "y", "value", "count", "notable_responses"]
}

RESPONSE_HEATMAP_SCHEMA = {
    "type": "object",
    "properties": {
        "x_axis": {
            "type": "array",
            "items": {"type": "string"}
        },
        "y_axis": {
            "type": "array",
            "items": {"type": "string"}
        },
        "cells": {
            "type": "array",
            "items": HEATMAP_CELL_SCHEMA
        },
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "description"]
            }
        },
        "hotspots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "insights": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["location", "insights"]
            }
        }
    },
    "required": ["x_axis", "y_axis", "cells", "patterns", "hotspots"]
}


