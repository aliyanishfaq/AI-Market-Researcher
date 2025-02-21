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
from openai import AsyncAzureOpenAI
from httpx import Timeout
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from SurveyTypes import SurveyRequest, Question, Option
import random
from llminference import LLMInference
from typing import Dict, Any, List
from survey_simulation import SurveySimulation, SimulationConfig
from personas import PersonaManager
from schema import PersonaType
from ask_endpoint.ask_prompts import AskPromptManager
from ask_endpoint.persona_loader import PersonaLoader
from deep_research.run import main as research_main

use_azure_openai = True
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

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

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
azure_openai_client = AsyncAzureOpenAI(
    api_key=azure_openai_api_key, 
    azure_endpoint = azure_openai_endpoint,
    api_version = "2024-08-01-preview",
)

prompt_manager = AskPromptManager()
persona_loader = PersonaLoader()

class QuestionRequest(BaseModel):
    persona_index: int
    question: str
    persona_type: PersonaType
    
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
    responses: List[Dict[str, Any]]  # Make this generic to accept any response structure
    question: str
    persona_type: PersonaType

class ResearchRequest(BaseModel):
    query: str
    breadth: int
    depth: int
    concurrency: int
    survey_results: Dict[str, Any]
    persona_responses: List[Dict[str, Any]]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
async def make_openai_request(prompt: str):
    if use_azure_openai:
        response = await azure_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
            seed=123
        )
    else:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
            seed=123
        )
    return response

@app.get("/")
async def root():
    return {"message": "Agent Colony is running"}


# Endpoint to Get All Personas
@app.get("/personas", response_model=List[dict])
async def get_persona(persona_type: PersonaType):
    """
    Returns the persona with the given index.
    """
    try:
        return persona_loader.get_personas(persona_type.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/research")
async def research(request: ResearchRequest):
    try:
        research_results = await research_main(
            query=request.query,
            breadth=request.breadth,
            depth=request.depth,
            concurrency=request.concurrency,
            service="azure",
            model="o3-mini",
            quiet=True,
            survey_results=request.survey_results,
            persona_responses=request.persona_responses
        )
        print(f"[research] results: {research_results}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing research: {str(e)}")
    return research_results


# Endpoint to Ask a Question to a Persona
@app.post("/ask")
async def ask_persona(request: QuestionRequest):
    """
    Handles user questions for a specific persona via OpenAI API.
    """
    try:
        # Get persona using PersonaLoader
        persona = persona_loader.get_persona(request.persona_type.value, request.persona_index)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        prompt = prompt_manager.format_prompt(request.persona_type, persona, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error formatting prompt: {str(e)}")
    
    # Send Request to OpenAI API
    try:
        response = await make_openai_request(prompt)
        answer = json.loads(response.choices[0].message.content.strip()) or {"error": "Failed to generate a response"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[ask][Exception] OpenAI API error: {str(e)}")
    
    print('[ask] answer: ', answer, 'persona: ', persona)
    return {"answer": answer, "persona": persona}


@app.post("/ask_survey_question")
async def ask_survey_question(request: QuestionRequest):
    """
    Handles the user question and then passes it to the survey endpoint.
    """
    try:
        prompt = f"""
        You are a survey expert. You are given a question. You job is to provide me with a list of options that are relevant to the question.
        The question is: {request.question}
        You can only provide upto 5 options.
        You need to provide me with the list in the following JSON format:
        {{
            "options": ["option1", "option2", "option3"]
        }}
        """
        response = await make_openai_request(prompt)
        options = json.loads(response.choices[0].message.content.strip()) or {"error": "Failed to generate a response"}
        
        # Create Question object with options from response
        question = Question(
            id="1",
            text=request.question,
            options=[Option(id=str(i+1), text=opt) for i, opt in enumerate(options["options"])]
        )
        
        survey_request = SurveyRequest(
            title=request.question,
            questions=[question],
            persona_type=request.persona_type,
            number_of_personas=32,
            number_of_samples=2000
        )
        return await run_survey(survey_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[ask_survey_question][Exception] Error: {str(e)}")


@app.post("/analyze_responses")
async def analyze_responses(request: ResponseAnalysisRequest):
    try:
        # Format responses based on persona type
        print(f"[analyze_responses] request: {request}")
        formatted_responses = []
        for i, resp in enumerate(request.responses):
            if request.persona_type == PersonaType.INTEL_EMPLOYEE:
                formatted_response = (
                    f"Response {i+1}: {resp.get('response')} "
                    f"Date: {resp.get('date')}"
                    f"(from {resp.get('name')}, {resp.get('role')}, {resp.get('location')}, "
                    f"Rating: {resp.get('rating')})"
                )
            elif request.persona_type == PersonaType.INTEL_PRODUCT_REVIEWER:
                formatted_response = (
                    f"Response {i+1}: {resp.get('response')} "
                    f"Date: {resp.get('date')}"
                    f"(Product: {resp.get('product_name')}, "
                    f"User: {resp.get('name')}, "
                    f"Technical Level: {resp.get('technical_level')})"
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported persona type: {request.persona_type}")
            
            formatted_responses.append(formatted_response)

        # Rest of the analysis logic remains the same
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
        print(f"[run_survey] params: {survey.persona_type}, {survey.number_of_personas}, {survey.number_of_samples}, {survey}")
        if survey.persona_type == PersonaType.INTEL_EMPLOYEE:
            persona_manager = PersonaManager(survey.persona_type)
        else:
            persona_manager = PersonaManager(survey.persona_type)

        llm = LLMInference(persona_manager)
        config = SimulationConfig(max_parallel_personas=3, thread_pool_size=2, timeout_seconds=300)

        async with SurveySimulation(llm, persona_manager, config, survey.number_of_personas, survey.number_of_samples, survey.persona_type) as simulation:
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
