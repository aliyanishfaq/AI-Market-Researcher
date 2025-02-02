from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import traceback
import time
from llminference import LLMInference
from schema import Persona
from response_analytics import QuestionAnalytics
from personas import PersonaManager
from SurveyTypes import Option
from survey_status import SimulationStatus, SurveyStage
from survery_meta_analysis import SurveyMetaAnalysis
import json

class SimulationConfig(BaseModel):
    """Configuration for the simulation"""
    max_parallel_personas: int = Field(default=16, description="Maximum number of personas to process in parallel")
    thread_pool_size: int = Field(default=16, description="Size of the thread pool for CPU-bound operations")
    timeout_seconds: int = Field(default=300, description="Timeout for each LLM request")


class SurveySimulation:
    """
    Orchestrates the simulation of a survey across multiple personas.
    
    This class manages:
    - Parallel processing of personas
    - Sequential processing of questions
    - Analytics generation
    - Error handling and status tracking
    """
    
    def __init__(self, llm: LLMInference, persona_manager: PersonaManager, config: SimulationConfig = SimulationConfig(), number_of_personas: int = 5, number_of_samples: int = 2000):
        self.llm = llm
        self.persona_manager = persona_manager
        self.personas = self.persona_manager.get_all_personas()
        self.config = config
        self.status = None
        self._executor = None
        self.number_of_personas = number_of_personas
        self.number_of_samples = number_of_samples
        
    async def __aenter__(self):
        """Setup for async context manager"""
        self._executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup for async context manager"""
        if self._executor:
            self._executor.shutdown(wait=True)

    async def _process_persona_question(self, persona: Persona, question_text: str, options: List[Option]) -> Dict[str, Any]:
        """Process a single question for a single persona"""
        start_time = time.time()
        try:
            options_text = [option.text for option in options]
            await asyncio.sleep(0.01)
            personality_summary_prompt = self.persona_manager.get_personality_summary_prompt(persona_id=persona.id)

            personality_summary = await self.llm.get_personality_summary(prompt=personality_summary_prompt)

            self.persona_manager.update_personality_summary(persona_id=persona.id, personality_summary=personality_summary)

            try:
                response = await self.llm.get_ensemble_distribution(persona=persona, question=question_text, options=options_text)
            except Exception as e:
                print(f"[SurveySimulation][_process_persona_question] Error: {str(e)}")
                response = None

            if response and response['relevant']:
                # Update conversation history
                conversation_summary = self.persona_manager.update_conversation_history(
                    persona_id=persona.id,
                    question=question_text,
                    distribution=response['option']
                )
                
                return {
                    "persona_id": persona.id,
                    "personality_summary": personality_summary,
                    "reliability_score": response['reliability_score'],
                    "distribution": response['option'],
                    "reason": response['reason'],
                    "error": None
                }
            
            return {
                "persona_id": persona.id,
                "personality_summary": personality_summary,
                "distribution": {},
                "reason": "Question not relevant for persona",
                "error": "Question not relevant for persona"
            }
            
        except Exception as e:
            return {
                "persona_id": persona.id,
                "personality_summary": "",
                "distribution": {},
                "reason": str(e),
                "error": str(e)
            }

    async def _process_question_batch(self, question_text: str, options: List[Option], batch: List[Persona]) -> List[Dict[str, Any]]:
        """Process a batch of personas for a single question"""
        start_time = time.time()
        tasks = [
            self._process_persona_question(persona, question_text, options)
            for persona in batch
        ]
        await asyncio.sleep(0.01)
        batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        return batch_responses

    async def _analyze_question_responses(self,all_responses: List[Dict[str, Any]], question: str, options: List[Option]) -> Dict[str, Any]:
        """Analyze responses for a question"""
        await asyncio.sleep(0.01)
        # Extract valid distributions
        options_text = [option.text for option in options]
        analytics = QuestionAnalytics(all_responses=all_responses, n_samples=self.number_of_samples)
        analysis = await analytics.analyze_survey_question(question=question, options=options_text)
        
        if asyncio.iscoroutine(analysis):
            print(f"[SurveySimulation][_analyze_question_responses] Warning: Analysis is a coroutine, expected a dictionary.")
        
        return analysis

    async def run_question(self, question_text: str, options: List[Option], question_index: int, total_questions: int) -> Dict[str, Any]:
        """Run a single question across all personas"""
        # Update status
        self.status.current_question = question_index + 1
        self.status.completed_personas = 0
        start_time = time.time()
        # Determine the number of personas to process
        num_personas = min(self.number_of_personas, len(self.personas))
        print(f"Number of personas: {num_personas}, total personas: {len(self.personas)}")
        await asyncio.sleep(0.01)
        # Create tasks for processing all batches concurrently
        batch_tasks = [
            self._process_question_batch(
                question_text,
                options,
                self.personas[i:i + self.config.max_parallel_personas]
            )
            for i in range(0, num_personas, self.config.max_parallel_personas)
        ]
        # Gather all batch responses concurrently
        all_batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
        # await asyncio.sleep(0.01)
        all_responses = []
        for batch_responses in all_batch_responses:
            if isinstance(batch_responses, Exception):
                print(f"[SurveySimulation][run_question] Error processing batch: {str(batch_responses)}")
                continue
            all_responses.extend(batch_responses)
            
            # Update completion status
            self.status.completed_personas += len(batch_responses)
            
            # Track errors
            for resp in batch_responses:
                if resp["error"]:
                    self.status.errors.append(
                        f"Error processing persona {resp['persona_id']} for question {question_text}: {resp['error']}"
                    )
                    self.status.completed_personas -= 1
        # Analyze responses
        await asyncio.sleep(0.01)
        analysis = await self._analyze_question_responses(all_responses, question_text, options)
        
        if asyncio.iscoroutine(analysis):
            print(f"[SurveySimulation][run_question] Warning: Analysis is a coroutine, expected a dictionary.")
        
        return analysis, self.status.completed_personas

    async def run_survey(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the complete survey simulation.
        """
        try:            
            start_time = time.time()
            # Initialize status tracking
            self.status = SimulationStatus(
                start_time=datetime.now(),
                current_question=0,
                total_questions=len(questions),
                completed_personas=0,
                total_personas=len(self.personas)
            )
            
            results = {}
            
            # Process questions concurrently
            question_tasks = [
                self.run_question(
                    question_text=question.text,
                    options=question.options,
                    question_index=i,
                    total_questions=len(questions)
                )
                for i, question in enumerate(questions)
            ]
            
            await asyncio.sleep(0.01)
            question_results = await asyncio.gather(*question_tasks, return_exceptions=True)
            # await asyncio.sleep(0.01)
            for i, result in enumerate(question_results):
                if isinstance(result, Exception):
                    print(f"[SurveySimulation][run_survey] Error processing question {i}: {str(result)}")
                    raise result
                elif asyncio.iscoroutine(result):
                    # Ensure coroutine is awaited
                    result = await result
                else:
                    question_id = questions[i].id
                    results[question_id], completed_personas = result
                    results[question_id]["completed_personas"] = completed_personas

            # completed personas for all questions
            completed_personas = {}
            for question_id, result in results.items():
                completed_personas[question_id] = result["completed_personas"]

            # Extract simplified response distributions for each question
            response_distributions = {}
            for question_id, result in results.items():
                if "basic_statistics" in result:
                    response_distributions[question_id] = result["basic_statistics"]["proportions"]

            final_result = {
                "question_results": results,
                "metadata": {
                    "total_personas": len(self.personas),
                    "total_questions": len(questions),
                    "error_count": len(self.status.errors),
                    "completed_personas": completed_personas,
                    "duration_seconds": (datetime.now() - self.status.start_time).total_seconds()
                }
            }

            await asyncio.sleep(0.01)
            self.status.update(stage=SurveyStage.COMPLETED, message="Survey completed")
            
            start_time = time.time()
            all_personas = self.persona_manager.get_all_personas()
            if asyncio.iscoroutine(all_personas):
                # Ensure coroutine is awaited
                all_personas = await all_personas
            survey_meta_analysis = SurveyMetaAnalysis(persona_data=all_personas, response_distributions=response_distributions)
            
            complete_analysis = await survey_meta_analysis.get_complete_analysis()
            await asyncio.sleep(0.01)
            final_result["complete_analysis"] = {
                "key_findings": complete_analysis.get("key_findings", []),
                "statistical_metrics": complete_analysis.get("statistical_metrics", {}),
                "recommendations": complete_analysis.get("recommendations", []),
                "alignment_analysis": complete_analysis.get("alignment_analysis", {}),
                "consistency_analysis": complete_analysis.get("consistency_analysis", {}),
                "demographic_insights": complete_analysis.get("demographic_insights", {})
            }

            return final_result
            
        except Exception as e:
            print(f"[SurveySimulation][run_survey] Fatal error: {str(e)}")
            raise
