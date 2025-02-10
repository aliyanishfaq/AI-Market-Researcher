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
from SurveyTypes import Question
from survey_meta_analysis.analysis_prompts import AnalysisPrompts
from schema import PersonaType
load_dotenv()

class SurveyMetaAnalysis:
    """
    Analyzes overall survey patterns and persona alignments across all questions.
    Uses Azure OpenAI API for generating structured insights about survey-wide patterns.
    """
    
    def __init__(self, persona_data: List[Dict[str, Any]], response_distributions: List[Dict[str, Any]], questions: List[Question], persona_type: PersonaType):
        """
        Initialize with survey responses and questions.
        
        Args:
            persona_data: List of persona responses with their conversation history
            response_distributions: List of response distributions for each question
            questions: List of questions asked in the survey
            persona_type: Type of persona being analyzed
        """
        self.persona_data = persona_data
        self.response_distributions = response_distributions
        self.questions = {q.id: q for q in questions}
        self.persona_type = persona_type
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')
        self.azure_openai_client = AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key, 
            azure_endpoint = self.azure_openai_endpoint,
            api_version = "2024-08-01-preview",
        )
        self.use_azure_openai = False

    async def analyze_persona_alignment(self) -> Dict[str, Any]:
        """Analyze how different persona types align in their responses."""
        response_data = self._format_persona_data()
        distribution_data = self._format_response_distributions()

        prompt = AnalysisPrompts.get_alignment_prompt(self.persona_type, response_data, distribution_data)
        print(f"[SurveyMetaAnalysis][analyze_persona_alignment] prompt: {prompt}")
        return await self._get_gemini_response(prompt)

    async def analyze_response_consistency(self) -> Dict[str, Any]:
        """Analyze how consistent personas are across different questions."""
        response_data = self._format_persona_data()
        distribution_data = self._format_response_distributions()

        prompt = AnalysisPrompts.get_consistency_prompt(self.persona_type, response_data, distribution_data)
        print(f"[SurveyMetaAnalysis][analyze_response_consistency] prompt: {prompt}")
        return await self._get_gemini_response(prompt)


    async def analyze_demographic_insights(self) -> Dict[str, Any]:
        """Generate insights about how different demographic groups respond."""
        response_data = self._format_persona_data()
        distribution_data = self._format_response_distributions()

        prompt = AnalysisPrompts.get_demographic_prompt(self.persona_type, response_data, distribution_data)
        print(f"[SurveyMetaAnalysis][analyze_demographic_insights] prompt: {prompt}")
        return await self._get_gemini_response(prompt)

    async def generate_key_findings(self, alignment_results: Dict[str, Any] = None, consistency_results: Dict[str, Any] = None, demographic_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate overall key findings from the survey."""
        start_time = time.time()
        
        # Run all analysis methods concurrently
        if not alignment_results:
            alignment_results = await self.analyze_persona_alignment()
        if not consistency_results:
            consistency_results = await self.analyze_response_consistency()
        if not demographic_results:
            demographic_results = await self.analyze_demographic_insights()


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
        print(f"[SurveyMetaAnalysis][generate_key_findings] prompt: {prompt}")
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
            question = self.questions.get(q_id)
            if question:
                dist_section += f"\nQuestion {q_id}: {question.text}\n"
            else:
                print(f"[SurveyMetaAnalysis][_format_response_distributions] Warning: Question with ID {q_id} not found")
            
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
            print(f"[SurveyMetaAnalysis][_get_gemini_response] Gemini API error: {str(e)}")
            return {"error": str(e)}

    async def get_complete_analysis(self) -> Dict[str, Any]:
        """Generate complete survey analysis including all aspects."""
        start_time = time.time()
        
        alignment_analysis, consistency_analysis, demographic_insights = await asyncio.gather(
            self.analyze_persona_alignment(),
            self.analyze_response_consistency(),
            self.analyze_demographic_insights()
        )

        key_findings = await self.generate_key_findings(alignment_analysis, consistency_analysis, demographic_insights)
        return {
            "key_findings": key_findings.get("primary_findings", []),
            "statistical_metrics": key_findings.get("statistical_metrics", {}),
            "recommendations": key_findings.get("recommendations", []),
            "alignment_analysis": alignment_analysis,
            "consistency_analysis": consistency_analysis,
            "demographic_insights": demographic_insights
        }
