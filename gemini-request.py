import google.generativeai as genai
from dotenv import load_dotenv
import os
import enum
import typing_extensions as typing

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Simple test schema
class TestSentiment(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"

class SimpleResponse(typing.TypedDict):
    sentiment: TestSentiment
    reason: str

def test_simple_schema():
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Convert to OpenAPI schema
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "mixed"]
            },
            "reason": {
                "type": "string"
            }
        },
        "required": ["sentiment", "reason"]
    }
    
    print("Testing with schema:", SimpleResponse)
    
    result = model.generate_content(
        "What's the sentiment of this text: 'I love this product!'",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=SimpleResponse
        )
    )
    print("Result:", result.text)

if __name__ == "__main__":
    test_simple_schema()