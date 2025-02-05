"""
[WARNING] THIS IS NOT WRITTEN IN A WAY THAT IT IS AGNOSTIC OF THE DATASET AND TYPE
LATER PROMPTS FOR EACH ANALYSIS CAN BE MADE MORE SPECIFIC TO THE DATASET

Each section is designed for specific UI visualizations:

Sentiment Flow

Sankey diagram or flow chart showing how sentiment moves across different aspects
Interactive paths showing sentiment journey
Color gradients for sentiment intensity


Personality Network

Interactive network graph
Nodes representing different persona types
Connections showing relationship strength
Size representing group size
Color intensity for sentiment


Response Heatmap

2D heatmap showing response patterns
Experience level vs Response options
Color intensity for frequency
Interactive tooltips for details


Theme Radar

Radar/Spider chart for themes
Each axis represents a theme
Area fill for strength
Color coding for sentiment
Interactive points for details


Persona Journey

Timeline/Journey map visualization
Multiple tracks for different persona types
Points showing key experience moments
Trend lines showing satisfaction over time
Interactive tooltips for key factors



Libraries:

D3.js for custom visualizations
recharts for React components
react-flow for network diagrams
nivo for rich charts
react-vis for heatmaps

"""

from typing import List, Dict, Any, Type
from pydantic import BaseModel
import numpy as np
from google.generativeai import GenerativeModel
import os
import google.generativeai as genai
from dotenv import load_dotenv
from schema import THEME_RADAR_SCHEMA, PERSONA_NETWORK_SCHEMA, SENTIMENT_FLOW_SCHEMA, RESPONSE_HEATMAP_SCHEMA
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import asyncio
import time 
from transform_schema import SchemaTransformer
from openai import AsyncAzureOpenAI


load_dotenv()

class QuestionQualitativeAnalysis:
    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = responses
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_client = AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key, 
            azure_endpoint = self.azure_openai_endpoint,
            api_version = "2024-08-01-preview",
        )
        self.use_azure_openai = True


    async def analyze_question(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Run all analyses concurrently and combine results"""
        start_time = time.time()
        theme_task = self._analyze_themes(question, options)
        network_task = self._analyze_network(question, options)
        sentiment_task = self._analyze_sentiment(question, options)
        pattern_task = self._analyze_patterns(question, options)

        theme_results, network_results, sentiment_results, pattern_results = await asyncio.gather(
            theme_task, network_task, sentiment_task, pattern_task
        )
        print(f"--- Time taken to analyze question: {time.time() - start_time}")
        return {
            "theme_analysis": theme_results,
            "network_analysis": network_results,
            "sentiment_analysis": sentiment_results,
            "response_patterns": pattern_results
        }

    async def _analyze_themes(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Analyze thematic elements with detailed theme extraction"""
        prompt = f"""
        Analyze the themes in these survey responses for: "{question}"
        Options: {', '.join(options)}
        Identify themes following these rules:
        1. Extract 3-5 major themes that appear across multiple responses
        2. For each theme:
           - Calculate strength based on frequency and emphasis in responses
           - Determine sentiment based on context and language
           - Find supporting quotes from the responses
           - Identify related themes
        3. Map connections between themes
        4. Identify the most significant theme based on impact and frequency
        
        Consider how different persona types discuss each theme.
        Look for both explicit and implicit theme mentions.
        
        Response Data:
        {self._format_for_theme_analysis()}
        """
        return await self._get_gemini_response(prompt, THEME_RADAR_SCHEMA)

    async def _analyze_network(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Analyze persona relationships and clusters"""
        prompt = f"""
        Analyze the relationships between personas for: "{question}"
        Options: {', '.join(options)}
        Create a persona network by:
        1. For each persona:
           - Identify role and experience level
           - Calculate sentiment score from responses
           - Extract key concerns and viewpoints
           - Note primary response choice
        2. Find connections between personas:
           - Identify shared viewpoints
           - Calculate similarity in responses
           - Note key differences
        3. Group personas into meaningful clusters
        
        Focus on both quantitative (response distributions) and 
        qualitative (reasoning, concerns) similarities.
        
        Response Data:
        {self._format_for_network_analysis()}
        """
        return await self._get_gemini_response(prompt, PERSONA_NETWORK_SCHEMA)

    async def _analyze_sentiment(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Analyze sentiment patterns and flow"""
        prompt = f"""
        Analyze sentiment flow in responses to: "{question}"
        Options: {', '.join(options)}
        Map sentiment progression by:
        1. Identify distinct stages in experiences
        2. For each stage:
           - Calculate positive/neutral/negative ratios
           - Identify key factors driving sentiment
           - Extract common phrases and expressions
        3. Determine overall trend
        4. Identify critical points where sentiment shifts
        
        Consider both:
        - Explicit sentiment statements
        - Implicit sentiment in language and reasoning
        
        Response Data:
        {self._format_for_sentiment_analysis()}
        """
        return await self._get_gemini_response(prompt, SENTIMENT_FLOW_SCHEMA)

    async def _analyze_patterns(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Analyze response patterns and create heatmap"""
        prompt = f"""
        Analyze response patterns for: "{question}"
        Options: {', '.join(options)}
        
        Create detailed response mapping:
        1. For each experience level and response option:
           - Calculate response frequency
           - Note significant patterns
           - Identify notable responses
        2. Find areas of high concentration
        3. Identify unexpected patterns
        4. Note experience-based trends
        
        Consider:
        - Role/experience influence on responses
        - Common reasoning patterns
        - Outlier responses
        
        Response Data:
        {self._format_for_pattern_analysis()}
        """
        return await self._get_gemini_response(prompt, RESPONSE_HEATMAP_SCHEMA)

    def _format_for_theme_analysis(self) -> str:
        """Format data optimized for theme extraction"""
        formatted_data = []
        for resp in self.responses:
            data = f"""
            Persona {resp['persona_id']}:
            Personality Summary: {resp['personality_summary']}
            Main Response: {max(resp['distribution'].items(), key=lambda x: x[1])[0]}
            Reasoning: {resp['reason']}
            """
            formatted_data.append(data)
        return "\n".join(formatted_data)

    def _format_for_network_analysis(self) -> str:
        """Format data optimized for network analysis"""
        formatted_data = []
        for resp in self.responses:
            # Extract key sentiments and preferences from personality summary
            data = f"""
            Persona {resp['persona_id']}:
            Personality Summary: {resp['personality_summary']}
            Response Distribution: {resp['distribution']}
            Reasoning: {resp['reason']}
            """
            formatted_data.append(data)
        return "\n".join(formatted_data)

    def _format_for_sentiment_analysis(self) -> str:
        """Format data optimized for sentiment analysis"""
        formatted_data = []
        for resp in self.responses:
            data = f"""
            Persona {resp['persona_id']}:
            Personality Summary: {resp['personality_summary']}
            Distribution: {resp['distribution']}
            Reasoning: {resp['reason']}
            """
            formatted_data.append(data)
        return "\n".join(formatted_data)

    def _format_for_pattern_analysis(self) -> str:
        """Format data optimized for pattern analysis"""
        formatted_data = []
        for resp in self.responses:
            data = f"""
            Persona {resp['persona_id']}:
            Response Pattern: {resp['distribution']}
            Reasoning: {resp['reason']}
            Personality Summary: {resp['personality_summary']}
            """
            formatted_data.append(data)
        return "\n".join(formatted_data)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_azure_openai_response(self, prompt: str, schema: Dict[str, Any]) -> Dict:
        """Get structured response from Azure OpenAI API."""
        schema_transformer = SchemaTransformer()
        wrapped_schema = schema_transformer.wrap_schema(schema, "theme", "Theme analysis schema", strict=True)
        try:
            response = await self.azure_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": wrapped_schema
                }
            )
            response_data = json.loads(response.choices[0].message.content)
            if isinstance(response_data, dict):
                return response_data
            else:
                return {"error": "Unexpected response type"}
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return {"error": str(e)}
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_gemini_response(self, prompt: str, schema: Dict[str, Any]) -> Dict:
        """Get structured response from Gemini"""
        try:
            if self.use_azure_openai:
                return await self._get_azure_openai_response(prompt, schema)
            else:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                    )
                )
                response_data = json.loads(response.text)
                if isinstance(response_data, dict):
                    return response_data
                else:
                    print("Unexpected response format, returning empty dictionary.")
                    return {}
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return {}  # Return empty dict for error case