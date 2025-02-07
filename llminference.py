from typing import List, Dict, Any
import httpx
import json
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from anthropic import AsyncAnthropicBedrock
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from openai import AsyncOpenAI
from openai import AsyncAzureOpenAI
import numpy as np
from personas import Persona
from personas import PersonaManager
import time
load_dotenv()

class LLMInference:
    def __init__(self, persona_manager: PersonaManager):
        self.persona_manager = persona_manager
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self.aws_access_key_id or not self.aws_secret_access_key or not self.aws_region:
            raise ValueError("AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION environment variables are required")
        
        self.temperatures = [0.1, 0.5, 1.0]  # Different temperatures for variation
        self.anthropic_client = AsyncAnthropicBedrock(
            aws_access_key=self.aws_access_key_id,
            aws_secret_key=self.aws_secret_access_key,
            aws_region=self.aws_region
        )
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        self.azure_openai_client = AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key, 
            azure_endpoint = self.azure_openai_endpoint,
            api_version = "2024-08-01-preview",
        )
        self.use_azure_openai = True
        # Add rate limiting properties
        self.last_request_time = None
        self._lock = asyncio.Lock()

    async def wait_for_cooldown(self):
        """Ensure at least 2 seconds between requests"""
        try:
            start_time = time.time()
            async with self._lock:
                now = datetime.now()
                if self.last_request_time:
                    elapsed = (now - self.last_request_time).total_seconds()
                    if elapsed < 2:
                        await asyncio.sleep(2 - elapsed)
                self.last_request_time = datetime.now()
                print(f"Time taken to wait for cooldown: {time.time() - start_time}")
        except Exception as e:
            print(f"[LLMInference][wait_for_cooldown] Error: {str(e)}")
            # Still update the time to prevent getting stuck
            self.last_request_time = datetime.now()

    async def _make_azure_openai_json_request(self, prompt: str, temperature: float, prompt_schema=None):
        response = await self.azure_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema", 
                "json_schema": prompt_schema
            },
        )
        return response
    
    async def _make_openai_json_request(self, prompt: str, temperature: float, prompt_schema=None):
        if self.use_azure_openai:
            response = await self._make_azure_openai_json_request(prompt, temperature, prompt_schema)
        else:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema", 
                    "json_schema": prompt_schema
                },
            )
        try:
            json_response = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[LLMInference][_make_openai_json_request] Error: {str(e)}")
            raise

        if not all(key in json_response for key in ['relevant', 'option', 'reason']):
            raise ValueError("Missing required fields in response")

        return json_response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=8, max=32), reraise=True)
    async def _make_azure_openai_request(self, prompt: str, temperature: float):
        response = await self.azure_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=8, max=32), reraise=True)
    async def _make_openai_request(self, prompt: str, temperature: float):
        if self.use_azure_openai:
            response = await self._make_azure_openai_request(prompt, temperature)   
        else:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        try:
            response = response.choices[0].message.content
        except Exception as e:
            print(f"[LLMInference][_make_openai_request] Error: {str(e)}")
            raise

        return response

    def _normalize_distribution(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """Normalize a probability distribution to ensure it sums to 1"""
        total = sum(distribution.values())
        if total == 0:
            return distribution
        return {k: float(v)/total for k, v in distribution.items()}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60), retry=retry_if_exception_type(Exception))
    async def get_distribution(self, prompt: str, temperature: float, prompt_schema=None) -> Dict[str, Any]:
        """Get probability distribution for options from LLM"""
        if prompt_schema:
            try:
                # await self.wait_for_cooldown()
                await asyncio.sleep(0.01)
                response = await self._make_openai_json_request(prompt, temperature, prompt_schema)

                # Convert array of objects into a dictionary if response is relevant
                if response.get("relevant"):
                    option_array = response.get("option", [])
                    option_dict = {item["option"]: item["probability"] for item in option_array}
                    response['option'] = option_dict

                # Normalize the option distribution if it exists
                if response.get('relevant') and 'option' in response:
                    response['option'] = self._normalize_distribution(response['option'])
                # print(f"Time taken to get distribution: {time.time() - start_time}")
                return response

            except Exception as e:
                print(f"[LLMInference][get_distribution] Error: {str(e)}")
                raise
        else:
            raise ValueError("Prompt schema is required")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=8, max=32), reraise=True)
    async def get_ensemble_distribution(self, persona: Persona, question: str, options: List[str]) -> Dict[str, Any]:
        """Get ensemble distribution by combining multiple calls with different temperatures"""
        distributions = []
        reasons = []  # Store reasons separately

        async def process_temperature(temp):
            try:
                await asyncio.sleep(0.01)
                prompt, prompt_schema = self.persona_manager.build_prompt(persona_id=persona.id, question=question, options=options)
                dist = await self.get_distribution(prompt, temp, prompt_schema)
                if not dist.get("relevant"):
                    return {
                        'relevant': False,
                        'option': {},
                        'reason': 'Persona not relevant to question'
                    }
                # Handle option_array as a dictionary
                option_dict = dist.get("option", {})
                if not isinstance(option_dict, dict):
                    print(f"[LLMInference][get_ensemble_distribution] Error: option_dict is not a dict. Actual type: {type(option_dict)}")
                    print(f"option_dict content: {option_dict}")
                    raise TypeError("option_dict is not a dict")
                
                distributions.append(option_dict)
                reasons.append(dist.get("reason", ""))
            except Exception as e:
                print(f"[LLMInference][get_ensemble_distribution] Error: {str(e)}")

        await asyncio.gather(*(process_temperature(temp) for temp in self.temperatures))

        # Add error handling for no distributions
        if not distributions:
            return {
                'relevant': False,
                'option': {},
                'reason': 'Failed to get distributions'
            }

        # options reliability
        option_variations = {}
        for option in options:
            probs = [dist.get(option, 0) for dist in distributions]
            mean_prob = np.mean(probs)
            if mean_prob != 0:
                cv = np.std(probs) / mean_prob # coefficient of variation
                option_variations[option] = cv
            
        mean_cv = np.mean(list(option_variations.values()))
        reliability_score = float(1 / (1 + mean_cv))

        reasons_string = "\n".join(reasons)
        # summarize reasons
        prompt = f"""
        You are part of a team that is analyzing responses from a survey.
        You are given a list of reasons why the user chose a particular options.
        You will be provided multiple reasons as to why an option was chosen.
        You will need to summarize the reason(s) into 20 words or less.

        Reasons: {reasons_string}
        """
        reason_summary = await self._make_openai_request(prompt, 0.2)

        # Only reaches here if all responses were relevant
        final_distribution = {
            'relevant': True,
            'option': {},  # Create proper option subdictionary
            'reason': reason_summary,  # Better reason formatting
            'reliability_score': reliability_score
        }
        
        # Fix: Get probability for each option from each distribution
        for option in options:
            option_probs = [dist.get(option, 0) for dist in distributions]
            final_distribution['option'][option] = sum(option_probs) / len(option_probs)
            
        return final_distribution

    async def get_personality_summary(self, prompt: str) -> str:
        """Get personality summary from LLM"""
        try:
            response = await self._make_openai_request(prompt, 0.2)

            return response
        except Exception as e:
            print(f"[LLMInference][get_personality_summary] Error: {str(e)}")
            raise


