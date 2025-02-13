import os
import json
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
load_dotenv()
from openai import AsyncAzureOpenAI

class QuestionClassifier:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = AsyncOpenAI(api_key=self.api_key)
        self.use_azure_openai = True

        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_client = AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key, 
            azure_endpoint = self.azure_openai_endpoint,
            api_version = "2024-08-01-preview",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60), retry=retry_if_exception_type(Exception))
    async def _make_azure_openai_request(self, prompt: str, temperature: float, schema=None):
        if schema is not None:
            response = await self.azure_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema", 
                    "json_schema": schema
                },
                seed=123
            ) 
        else:
            response = await self.azure_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
                seed=123
            )
        return response
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60), retry=retry_if_exception_type(Exception))
    async def _make_openai_request(self, prompt: str, temperature: float, schema=None):
        if self.use_azure_openai:
            response = await self._make_azure_openai_request(prompt, temperature, schema)
        else:
            if schema is not None:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema", 
                        "json_schema": schema
                    },
                    seed=123
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    seed=123
                )
        try:
            json_response = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[QuestionClassifier][_make_openai_request] Error: {str(e)}")
            raise

        if not all(key in json_response for key in ['scale_type', 'is_likert', 'ordered_options']):
            print(f"[QuestionClassifier][_make_openai_request] Error: Missing required fields in response: {json_response}")
            raise ValueError("Missing required fields in response")

        return json_response

    async def classify(self, question: str, options: List[str]) -> Dict[str, Any]:
        prompt = f"""
        Analyze these survey question options and determine their type and structure:
        Question: {question}
        Options: {', '.join(options)}

        Consider:
        1. Is this a Likert scale? (agreement, satisfaction, frequency, etc.)
        2. Is there a natural ordering to the options?
        3. Is it a binary choice?
        4. Is it a categorical/nominal selection?

        Note the examples below:
        Likert scale: "Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"
        Binary choice: "Yes", "No"
        Categorical: "Red", "Blue", "Green"

        Return your analysis in the following JSON format:
        {{
            "scale_type": "string (likert, numeric, binary, or categorical)",
            "is_likert": "boolean",
            "ordered_options": "list of options in order (if applicable and the order should be NEGATIVE to POSITIVE), null if no natural order",
        }}
        """

        schema = {
            "name": "survey_analysis",
            "description": "Analysis of survey question options to determine type and structure",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scale_type": {
                        "type": "string",
                        "description": "Type of scale: likert, numeric, binary, or categorical"
                    },
                    "is_likert": {
                        "type": "boolean",
                        "description": "True if the scale is a Likert scale"
                    },
                    "ordered_options": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "string"
                        },
                        "description": "List of options ordered from NEGATIVE to POSITIVE, or null if no natural order"
                    }
                },
                "required": ["scale_type", "is_likert", "ordered_options"],
                "additionalProperties": False
            }
        }

        return await self._make_openai_request(prompt, 0.2, schema=schema)
