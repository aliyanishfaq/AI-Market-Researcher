"""
This file is used to analyze the survey data and generate insights about the survey.
ROUGH CODE: NOT VERIFIED FOR USEFULNESS
The API endpoint is changed to Azure OpenAI API from Gemini API.
"""


import os
import json
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from schema import Persona
import time
import asyncio
from openai import AsyncAzureOpenAI
load_dotenv()

class SurveyMetaAnalysis:
    """
    Analyzes overall survey patterns and persona alignments across all questions.
    Uses Azure OpenAI API for generating structured insights about survey-wide patterns.
    """
    
    def __init__(self, persona_data: List[Dict[str, Any]], response_distributions: List[Dict[str, Any]]):
        """
        Initialize with survey responses and questions.
        
        Args:
            responses: List of persona responses with their conversation history
            questions: List of questions asked in the survey
        """
        self.persona_data = persona_data
        self.response_distributions = response_distributions
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.azure_openai_client = AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key, 
            azure_endpoint = self.azure_openai_endpoint,
            api_version = "2024-08-01-preview",
        )
        self.use_azure_openai = True

    async def analyze_persona_alignment(self) -> Dict[str, Any]:
        """Analyze how different persona types align in their responses."""
        prompt = f"""
        Analyze the alignment patterns between different personas in these survey responses.
        Focus on identifying groups of personas that show similar response patterns.

        Survey Questions:
        {self._format_response_distributions()}

        Response Data:
        {self._format_persona_data()}

        Generate insights about:
        1. Which groups of personas show similar response patterns
        2. The strength of alignment within each group
        3. Any notable outliers or divergent groups
        4. Factors that might explain these alignments
        
        Return a JSON object with:
        {{
            "most_aligned_group": {{
                "group_name": string,
                "alignment_score": number,
                "member_count": number,
                "key_characteristics": List[string]
            }},
            "alignment_patterns": [
                {{
                    "group": string,
                    "score": number (0-1),
                    "size": number,
                    "common_traits": List[string]
                }}
            ],
            "notable_outliers": [
                {{
                    "description": string,
                    "reason": string
                }}
            ]
        }}
        """
        return await self._get_gemini_response(prompt)

    async def analyze_response_consistency(self) -> Dict[str, Any]:
        """Analyze how consistent personas are across different questions."""
        prompt = f"""
        Analyze the consistency of responses across different questions for each persona.
        Look for patterns in how responses change or remain stable.

        Survey Questions:
        {self._format_response_distributions()}

        Response Data:
        {self._format_persona_data()}

        Generate insights about:
        1. Overall response consistency across questions
        2. How different persona types vary in consistency
        3. Any patterns in response changes
        4. Factors that might influence consistency

        Return a JSON object with:
        {{
            "overall_consistency": {{
                "score": number (0-1),
                "confidence_level": number (0-1),
                "influential_factors": List[string]
            }},
            "consistency_by_group": [
                {{
                    "group": string,
                    "consistency_score": number (0-1),
                    "pattern_description": string
                }}
            ],
            "response_trends": [
                {{
                    "trend_description": string,
                    "affected_groups": List[string],
                    "significance": number (0-1)
                }}
            ]
        }}
        """
        return await self._get_gemini_response(prompt)

    async def analyze_demographic_insights(self) -> Dict[str, Any]:
        """Generate insights about how different demographic groups respond."""
        prompt = f"""
        Analyze how different demographic and role-based groups respond to the survey.
        Look for patterns based on experience level, role type, and other demographics.

        Survey Questions:
        {self._format_response_distributions()}

        Response Data:
        {self._format_persona_data()}

        Generate insights about:
        1. How different role types typically respond
        2. Impact of experience level on responses
        3. Any location-based patterns
        4. Notable demographic correlations

        Return a JSON object with:
        {{
            "role_based_insights": [
                {{
                    "role_type": string,
                    "key_patterns": List[string],
                    "sentiment_score": number (0-1)
                }}
            ],
            "experience_level_insights": [
                {{
                    "level": string,
                    "typical_responses": string,
                    "significant_differences": List[string]
                }}
            ],
            "demographic_correlations": [
                {{
                    "factor": string,
                    "correlation_strength": number (0-1),
                    "description": string
                }}
            ]
        }}
        """
        return await self._get_gemini_response(prompt)

    async def generate_key_findings(self) -> Dict[str, Any]:
        """Generate overall key findings from the survey."""
        start_time = time.time()
        
        # Run all analysis methods concurrently
        alignment_results, consistency_results, demographic_results = await asyncio.gather(
            self.analyze_persona_alignment(),
            self.analyze_response_consistency(),
            self.analyze_demographic_insights()
        )

        prompt = f"""
        Generate key findings based on the following analysis results:
        
        Alignment Analysis: {json.dumps(alignment_results)}
        Consistency Analysis: {json.dumps(consistency_results)}
        Demographic Analysis: {json.dumps(demographic_results)}

        Generate a concise summary of the most important findings.

        Return a JSON object with:
        {{
            "primary_findings": [
                {{
                    "title": string,
                    "description": string,
                    "significance": number (0-1),
                    "supporting_data": string
                }}
            ],
            "statistical_metrics": {{
                "confidence_level": number (0-1),
                "margin_of_error": number (0-1),
                "response_quality_score": number (0-1)
            }},
            "recommendations": [
                {{
                    "title": string,
                    "description": string,
                    "priority": number (0-1)
                }}
            ]
        }}
        """
        return await self._get_gemini_response(prompt)

    def _format_persona_data(self) -> str:
        """Format response data for prompt inclusion."""
        formatted_data = []
        for persona in self.persona_data:
            data = f"""
            Persona Description:
            Name: {persona.name}
            Role: {persona.role}
            Personality: {persona.personality_summary}
            
            Responses:
            """
            
            for response in persona.conversation_history:
                data += f"\n- {response['question']}\n  {response['summary']}"
            
            formatted_data.append(data)
        
        # Add response distributions section
        dist_section = "\nOverall Response Distributions:\n"
        for q_id, distributions in self.response_distributions.items():
            dist_section += f"\nQuestion {q_id}:\n"
            for sentiment, percentage in distributions.items():
                dist_section += f"- {sentiment}: {percentage:.1%}\n"
        
        formatted_data.append(dist_section)
        
        return "\n---\n".join(formatted_data)

    def _format_response_distributions(self) -> str:
        """
        Format response distributions data into a readable string format.
        Handles any response options dynamically.
        
        Returns:
            str: Formatted string showing distribution percentages for each question
        """
        dist_section = "Response Distributions Analysis:\n"
        
        for q_id, distributions in self.response_distributions.items():
            dist_section += f"\nQuestion {q_id}:\n"
            
            # Simply display whatever options and percentages exist in the data
            for response_option, percentage in distributions.items():
                dist_section += f"- {response_option}: {percentage:.1%}\n"
        
        return dist_section

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_azure_openai_response(self, prompt: str) -> Dict[str, Any]:
        """Get structured response from Gemini API."""
        try:
            response = await self.azure_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            response_data = json.loads(response.choices[0].message.content)
            if isinstance(response_data, dict):
                return response_data
            else:
                print(f"[SurveyMetaAnalysis][_get_azure_openai_response] response_data is not a dictionary: {response_data}")
                return {"error": "Unexpected response type"}
        except Exception as e:
            return {"[SurveyMetaAnalysis][_get_azure_openai_response] error": str(e)}
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_gemini_response(self, prompt: str) -> Dict[str, Any]:
        """Get structured response from Gemini API."""
        try:
            if self.use_azure_openai:
                return await self._get_azure_openai_response(prompt)
            else:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
                response_data = json.loads(response.text)
                if isinstance(response_data, dict):
                    return response_data
                else:
                    return {"error": "Unexpected response type"}
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return {"error": str(e)}

    async def get_complete_analysis(self) -> Dict[str, Any]:
        """Generate complete survey analysis including all aspects."""
        start_time = time.time()
        
        # Run all analysis methods concurrently
        key_findings, alignment_analysis, consistency_analysis, demographic_insights = await asyncio.gather(
            self.generate_key_findings(),
            self.analyze_persona_alignment(),
            self.analyze_response_consistency(),
            self.analyze_demographic_insights()
        )
        return {
            "key_findings": key_findings.get("primary_findings", []),
            "statistical_metrics": key_findings.get("statistical_metrics", {}),
            "recommendations": key_findings.get("recommendations", []),
            "alignment_analysis": alignment_analysis,
            "consistency_analysis": consistency_analysis,
            "demographic_insights": demographic_insights
        }
