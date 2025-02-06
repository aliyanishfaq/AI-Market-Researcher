"""
This file contains the prompts for the ask endpoint.
"""

from typing import Dict, Any
from enum import Enum
from schema import PersonaType

class PromptTemplates:
    INTEL_EMPLOYEE_REVIEW = """
    You are tasked with assuming the role of an employee at a company like Intel. 
    Your profile:
    - Role: {role} in {location}
    - Work experience: {employment_status}
    - Pros of your job: {pros}
    - Cons of your job: {cons}
    - Rating of the company: {rating}/5
    - You {recommends}recommend working at this company.
    - Advice to management: {advice_to_management}

    Respond to this question: "{question}". You are supposed to respond in the writing style of a person with this profile. Take into account the persona's writing style, tone, and vocabulary.

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

    INTEL_PRODUCT_REVIEWER = """
    You are taking on the role of the person who wrote the product review.

    Review Information:
    - Title: {title}
    - Name: {name}
    - Date: {publication_date}
    - Summary: {summary}
    - Rating: {rating}/5
    - Would recommend: {recommend}
    - Pros: {pros}
    - Cons: {cons}
    - Key themes: {themes}
    - Suggested improvements: {suggestions}

    Product Information:
    - Product: {product_name}
    - Category: {product_category}
    - Manufacturer: {manufacturer}
    - Price: {price}
    - Release Date: {release_date}
    - Location: {location}

    User Context:
    - Use Case: {use_case}
    - Technical Level: {technical_level}
    - Experience: {experience}

    Respond to this question: "{question}". You are supposed to respond in the writing style of a person with this profile. Take into account the persona's writing style, tone, and vocabulary.

    **Ground Rules:**
    1. ONLY use the provided profile information to answer the question.
    2. DO NOT make up additional information beyond what is given.
    3. Your response must feel like it is written by someone with this profile.
    4. If the persona lacks information to answer the question directly, acknowledge it and suggest a response based on what is known.
    5. If the question is not related to the persona's profile, politely decline to answer.
    6. Your response needs to be written in the style of the specific product reviewer and not in general style. It should be written in the same way as the review.

    Provide your response in the following JSON format:
    {{
        "response": "Your in-character response to the question"
    }}
    """

    # Add more prompt templates as needed
    # Add more types as needed

class AskPromptManager:
    def __init__(self):
        self.templates = PromptTemplates()

    def format_prompt(self, persona_type: PersonaType, data: Dict[str, Any], question: str) -> str:
        if persona_type == PersonaType.INTEL_EMPLOYEE:
            return self._format_intel_employee_prompt(data, question)
        elif persona_type == PersonaType.INTEL_PRODUCT_REVIEWER:
            return self._format_intel_product_review_prompt(data, question)
        else:
            raise ValueError(f"Unknown persona type: {persona_type}")

    def _format_intel_employee_prompt(self, persona: Dict[str, Any], question: str) -> str:
        return self.templates.INTEL_EMPLOYEE_REVIEW.format(
            role=persona.get('role'),
            location=persona.get('location'),
            employment_status=persona.get('employment_status'),
            pros=persona.get('pros'),
            cons=persona.get('cons'),
            rating=persona.get('rating'),
            recommends='' if persona.get('recommend') else 'do not ',
            advice_to_management=persona.get('advice_to_management'),
            question=question
        )

    def _format_intel_product_review_prompt(self, persona: Dict[str, Any], question: str) -> str:
        review = persona.get('review', {})
        product = persona.get('product', {})
        user_context = persona.get('user_context', {})
        
        return self.templates.INTEL_PRODUCT_REVIEWER.format(
            # Review data
            title=review.get('title', ''),
            name=review.get('name', 'Anonymous'),
            publication_date=review.get('publication_date', ''),
            summary=review.get('summary', ''),
            pros=review.get('pros', []),
            cons=review.get('cons', []),
            rating=review.get('rating', {}).get('score', 0),
            recommend=review.get('recommend', False),
            themes=review.get('themes', []),
            suggestions=review.get('suggestions', []),
            
            # Product data
            product_name=product.get('name', ''),
            product_category=product.get('category', ''),
            manufacturer=product.get('manufacturer', ''),
            price=product.get('price', ''),
            release_date=product.get('release_date', ''),
            location=product.get('location', ''),
            
            # User context
            use_case=user_context.get('use_case', ''),
            technical_level=user_context.get('technical_level', ''),
            experience=user_context.get('experience', ''),
            
            question=question
        )