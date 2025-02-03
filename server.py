from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from typing import List
from fastapi import HTTPException
from openai import AsyncOpenAI
from httpx import Timeout
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from SurveyTypes import SurveyRequest
import random
from llminference import LLMInference
from typing import Dict, Any, List
from survey_simulation import SurveySimulation, SimulationConfig
from personas import PersonaManager

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

with open('glassdoor.json', 'r') as f:
    PERSONAS = json.load(f)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

class QuestionRequest(BaseModel):
    persona_index: int
    question: str

class Response(BaseModel):
    response: str
    name: str
    role: str
    rating: float
    location: str
    recommends: bool
    experience: str
    date: str
    advice_to_management: str

class ResponseAnalysisRequest(BaseModel):
    responses: List[Response]
    question: str

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
async def make_openai_request(prompt: str):
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    return response

@app.get("/")
async def root():
    return {"message": "Agent Colony is running"}


# Endpoint to Get All Personas
@app.get("/personas", response_model=List[dict])
async def get_persona(persona_index: int):
    """
    Returns the persona with the given index.
    """
    return PERSONAS[persona_index] or {"error": "Persona not found"}


# Endpoint to Ask a Question to a Persona
@app.post("/ask")
async def ask_persona(request: QuestionRequest):
    """
    Handles user questions for a specific persona via OpenAI API.
    """
    # Validate Persona Index
    if request.persona_index < 0 or request.persona_index >= len(PERSONAS):
        raise HTTPException(status_code=404, detail="Persona not found")

    # Fetch the Persona
    persona = PERSONAS[request.persona_index]

    # Construct Prompt for OpenAI
    prompt = f"""
    You are tasked with assuming the role of an employee at a company like Intel. 
    Your profile:
    - Role: {persona.get('role')} in {persona.get('location')}
    - Work experience: {persona.get('employment_status')}
    - Pros of your job: {persona.get('pros')}
    - Cons of your job: {persona.get('cons')}
    - Rating of the company: {persona.get('rating')}/5
    - You {'' if persona.get('recommend') else 'do not '}recommend working at this company.
    - Advice to management: {persona.get('advice_to_management')}

    Respond to this question: "{request.question}". You are supposed to respond in the writing style of a person with this profile. Take into account the persona's writing style, tone, and vocabulary.

    **Ground Rules:**
    1. ONLY use the provided profile information to answer the question.
    2. DO NOT make up additional information beyond what is given.
    3. Your response must feel like it is written by someone with this profile.
    4. If the persona lacks information to answer the question directly, acknowledge it and suggest a response based on what is known.
    5. If the question is not related to the persona's profile, politely decline to answer.
    Provide your response in the following JSON format:
    {{
        "response": "Your in-character response to the question"
    }}
    """

    # Send Request to OpenAI API
    try:
        response = await make_openai_request(prompt)
        answer = json.loads(response.choices[0].message.content.strip()) or {"error": "Failed to generate a response"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[ask][Exception] OpenAI API error: {str(e)}")
    
    print('[ask] answer: ', answer, 'persona: ', persona)
    return {"answer": answer, "persona": persona}


@app.post("/analyze_responses")
async def analyze_responses(request: ResponseAnalysisRequest):
    try:
        # Format responses correctly using dot notation
        formatted_responses = [
            f"Response {i+1}: {resp.response} (from {resp.name}, {resp.role}, {resp.date})" 
            for i, resp in enumerate(request.responses)
        ]
        
        analysis_prompt = (
            f"Analyze these responses to: \"{request.question}\"\n\n"
            f"Responses with their timestamps and details:\n"
            f"{chr(10).join(formatted_responses)}\n\n"
            f"Provide a comprehensive temporal analysis in this exact JSON format:\n"
            "{\n"
            '    "sentimentTimeSeries": [\n'
            "        { \n"
            '            "date": "timestamp",\n'
            '            "positive": "percentage (0-100)",\n'
            '            "negative": "percentage (0-100)"\n'
            "        }\n"
            "    ],\n"
            '    "themeDistribution": [\n'
            "        {\n"
            '            "theme": "theme name",\n'
            '            "2022": "value (0-100)",\n'
            '            "2023": "value (0-100)",\n'
            '            "2024": "value (0-100)"\n'
            "        }\n"
            "    ],\n"
            '    "emotionAnalysis": [\n'
            "        {\n"
            '            "emotion": "emotion name",\n'
            '            "2022": "intensity (0-100)",\n'
            '            "2023": "intensity (0-100)",\n'
            '            "2024": "intensity (0-100)"\n'
            "        }\n"
            "    ],\n"
            '    "insights": [\n'
            "        {\n"
            '            "title": "Key insight title",\n'
            '            "description": "Detailed description",\n'
            '            "type": "sentiment|themes|improvement"\n'
            "        }\n"
            "    ]\n"
            "}"
        )

        # Generate analysis
        response = await make_openai_request(analysis_prompt)
        # Extract content from OpenAI response correctly
        analysis_response = response.choices[0].message.content
        
        try:
            parsed_analysis = json.loads(analysis_response)
            
            structured_response = {
                "sentimentTimeSeries": parsed_analysis.get("sentimentTimeSeries", []),
                "themeDistribution": parsed_analysis.get("themeDistribution", []),
                "emotionAnalysis": parsed_analysis.get("emotionAnalysis", []),
                "insights": parsed_analysis.get("insights", [])
            }
            
            return structured_response

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {analysis_response}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse analysis response: {str(e)}"
            )

    except Exception as e:
        print(f"Analysis failed with error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )



"""
Runs a survey across all personas and analyzes the results.

Survey flow:
1. Process each questions sequentially to maintain conversation history
2. 

This endpoint:
    1. Processes questions sequentially to maintain conversation history
    2. For each question, processes all personas in parallel
    3. Aggregates and analyzes results for each question before moving to the next
"""
@app.post("/survey/run")
async def run_survey(survey: SurveyRequest) -> Dict[str, Any]:
    try:
        print(f"[run_survey] params: {survey.data_source}, {survey.number_of_personas}, {survey.number_of_samples}, {survey}")
        if survey.data_source == "glassdoor":
            persona_manager = PersonaManager(survey.data_source)
        else:
            persona_manager = PersonaManager(survey.data_source)

        llm = LLMInference(persona_manager)
        config = SimulationConfig(max_parallel_personas=3, thread_pool_size=2, timeout_seconds=300)

        async with SurveySimulation(llm, persona_manager, config, survey.number_of_personas, survey.number_of_samples) as simulation:
            results = await simulation.run_survey(survey.questions)
        print(f"[run_survey] results: {results}")
        return results
        
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing survey: {str(e)}"
        )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) # change to False in production
