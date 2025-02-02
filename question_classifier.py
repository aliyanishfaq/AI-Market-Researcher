import os
import json
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class QuestionClassifier:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = AsyncOpenAI(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60), retry=retry_if_exception_type(Exception))
    async def _make_openai_request(self, prompt: str, temperature: float):
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        try:
            json_response = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[QuestionClassifier][_make_openai_request] Error: {str(e)}")
            raise

        if not all(key in json_response for key in ['scale_type', 'is_likert', 'ordered_options']):
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
        return await self._make_openai_request(prompt, 0.2)