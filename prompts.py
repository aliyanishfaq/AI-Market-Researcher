from schema import Persona
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field

def build_employee_prompt_v1(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Generate a survey simulation prompt for an employee profile."""
    prompt = f"""You are a company survey response predictor. 
    Your task is to estimate realistic probability distributions for how an employee with the following profile might respond, acknowledging that even predictable employees can vary their responses due to recent events or changes in mood.

    Employee profile:
    - Job role: {persona.role} based in {persona.location}
    - Employment details: {persona.employment_status}
    - Positive aspects of their role: {persona.pros}
    - Challenges or negatives of their role: {persona.cons}
    - Company rating: {persona.rating}/5
    - Overall attitude: {'Likely to recommend' if persona.recommend else 'Unlikely to recommend'}, {'Approves of CEO' if persona.ceo_approval else 'unlikely to approve of CEO'}, {'Optimistic' if persona.business_outlook else 'pessimistic'} outlook on company performance
    - Core concerns: {persona.advice_to_management}

    Context from previous interactions:
    """
    if persona.conversation_history:
        prompt += "The individual has previously answered the following questions:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Based on this information, predict the probability distribution for how this employee would answer the question: "{question}"

    Account for:
    - Variations in daily experiences and emotions
    - Their earlier feedback and attitudes
    - How frustrations and benefits might influence their response
    - Patterns reflected in their sentiment and ratings

    Provide probabilities for each response option, ensuring they sum to 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Return your response in this JSON format:
    {
        "relevant": boolean (true if the question is related to the profile, false otherwise),
        "option": [
            {
                "option": "option1",
                "probability": probability
            },
            {
                "option": "option2",
                "probability": probability
            },
            ...
        ],
        "reason": "Detailed reasoning for the assigned probabilities based on the persona"
    }
    
    If the question is unrelated to the employee's experience, set "relevant": false and leave other fields as empty strings.
    """

    schema = {
        "name": "employee_response_v1",
        "description": "Simulated employee survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the employee's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_employee_prompt_v2(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Create a tailored prompt to simulate employee survey responses."""
    prompt = f"""You are an AI model tasked with simulating employee survey responses. 
    Your objective is to generate probability distributions for each option based on the following employee's profile, recognizing that responses may vary depending on recent experiences or emotions.

    Employee details:
    - Job title: {persona.role}, located in {persona.location}
    - Employment type: {persona.employment_status}
    - Highlights of the job: {persona.pros}
    - Pain points of the job: {persona.cons}
    - Company rating: {persona.rating}/5
    - Sentiment summary: {'Recommends' if persona.recommend else 'Does not recommend'}, {'Approves of CEO' if persona.ceo_approval else 'does not approve of CEO'}, {'Positive' if persona.business_outlook else 'poor'} business outlook
    - Major concerns or feedback: {persona.advice_to_management}

    Previous survey responses:
    """
    if persona.conversation_history:
        prompt += "The employee has previously responded as follows:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Given this context, simulate the probability distribution for their answer to the question: "{question}"

    Consider the following factors:
    - Daily fluctuations in their mood or recent experiences
    - Feedback and sentiment patterns from their previous interactions
    - Key challenges and benefits identified in their role
    - General tendencies based on their sentiment and ratings

    List probabilities for each option below, ensuring they total 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Provide a response in the following JSON format:
    {
        "relevant": boolean (true if the question aligns with their profile, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "A clear explanation of how the persona's profile informed the distribution"
    }
    
    If the question does not align with the persona, set "relevant": false and leave the other fields blank.
    """

    schema = {
        "name": "employee_response_v2",
        "description": "Simulated employee survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the employee's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_employee_prompt_v3(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Generate a probability-based survey response prediction prompt."""
    prompt = f"""You are a simulation model for employee surveys. 
    Your role is to predict the probability distribution of responses an employee might give, considering the nuances of their profile and the potential for variation influenced by recent experiences or emotional states.

    Employee profile summary:
    - Position: {persona.role} located in {persona.location}
    - Employment details: {persona.employment_status}
    - Positive job attributes: {persona.pros}
    - Negative job attributes: {persona.cons}
    - Overall company rating: {persona.rating}/5
    - Sentiment analysis: {'Likely to recommend' if persona.recommend else 'Not likely to recommend'}, {'Positive CEO approval' if persona.ceo_approval else 'negative CEO approval'}, {'Positive business outlook' if persona.business_outlook else 'poor business outlook'}
    - Primary concerns: {persona.advice_to_management}

    Historical responses:
    """
    if persona.conversation_history:
        prompt += "The following context is derived from their earlier responses:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Based on the above, estimate the likelihood of their response to the question: "{question}"

    Take into account:
    - Daily mood shifts and relevant interactions
    - Patterns observed in their feedback and attitudes
    - Highlights and challenges within their job role
    - Their typical rating and sentiment trends

    Specify the probabilities for each option, ensuring the total equals 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Deliver your output as a JSON object with the following structure:
    {
        "relevant": boolean (true if the question applies to the persona, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "Explanation of the assigned probabilities based on the persona's attributes"
    }
    
    If the question is irrelevant, set "relevant": false and leave the other fields blank.
    """

    schema = {
        "name": "employee_response_v3",
        "description": "Simulated employee survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the employee's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_employee_prompt_v4(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Build a prompt of employee for the LLM including persona context and conversation history"""
    prompt = f"""You are a survey response simulator for company surveys. 
    Your task is to generate realistic probability distributions for how an employee with this profile would respond, considering that even consistent employees might occasionally give different responses depending on their recent experiences and mood.

    Consider the following employee profile:
    - Role: {persona.role} in {persona.location}
    - Work experience: {persona.employment_status}
    - Pros of your job: {persona.pros}
    - Cons of your job: {persona.cons}
    - Rating of the company: {persona.rating}/5
    - Overall sentiment: {'' if persona.recommend else 'Does not recommend'}, {'' if persona.ceo_approval else 'does not approve of CEO'}, {'' if persona.business_outlook else 'negative'} business outlook
    - Key concerns: {persona.advice_to_management}

    Previous conversation context:
    """
    if persona.conversation_history:
        prompt += "The person responded to the following questions with following answers:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Based on this profile, simulate the probability distribution for how this employee would respond to: "{question}"

    Consider:
    - Day-to-day variations in mood and experiences
    - Recent interactions reflected in their review
    - Impact of mentioned frustrations and positive points
    - Overall sentiment and rating tendency
    
    Provide probabilities for each option, ensuring they sum to 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Return a JSON object with:
    {
        "relevant": boolean (true if question relates to their experience, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "Detailed explanation of why this distribution makes sense for this persona"
    }
    
    If the question is not relevant to the persona's experience, set "relevant": false and leave other fields as empty strings.
    """

    schema = {
        "name": "employee_response_v4",
        "description": "Simulated employee survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the employee's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_product_reviewer_prompt_v1(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Generate a survey simulation prompt for a product reviewer profile."""
    prompt = f"""You are a product survey response predictor. 
    Your task is to estimate realistic probability distributions for how a customer with the following profile might respond, acknowledging that even satisfied customers can vary their responses based on recent experiences and product usage.

    Customer profile:
    - Product: {persona.product_name} ({persona.product_category})
    - Location: {persona.location}
    - Rating given: {persona.rating}/5
    - Review title: {persona.title}
    - Positive aspects: {', '.join(persona.pros) if isinstance(persona.pros, list) else persona.pros}
    - Negative aspects: {', '.join(persona.cons) if isinstance(persona.cons, list) else persona.cons}
    - Key themes: {', '.join(persona.themes) if persona.themes else 'None specified'}
    - Overall attitude: {'Recommends product' if persona.recommend else 'Does not recommend product'}
    - Usage context: {persona.use_case}
    - Technical expertise: {persona.technical_level}

    Context from previous interactions:
    """
    if persona.conversation_history:
        prompt += "The customer has previously answered the following questions:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Based on this information, predict the probability distribution for how this customer would answer the question: "{question}"

    Account for:
    - Their overall product satisfaction level
    - Specific experiences mentioned in their review
    - Technical background and use case context
    - Pattern of likes and dislikes

    Provide probabilities for each response option, ensuring they sum to 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Return your response in this JSON format:
    {
        "relevant": boolean (true if the question relates to their experience, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "Detailed reasoning for the assigned probabilities based on the persona"
    }
    
    If the question is unrelated to the customer's experience, set "relevant": false and leave other fields as empty strings.
    """

    schema = {
        "name": "product_reviewer_response_v1",
        "description": "Simulated product reviewer survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the customer's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_product_reviewer_prompt_v2(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Create a tailored prompt to simulate product reviewer survey responses."""
    prompt = f"""You are an AI model tasked with simulating product review survey responses. 
    Your objective is to generate probability distributions for each option based on the following customer's profile, recognizing that responses may vary depending on usage patterns and experiences.

    Product review details:
    - Product details: {persona.product_name} by {persona.manufacturer}
    - Category: {persona.product_category}
    - Customer location: {persona.location}
    - Satisfaction rating: {persona.rating}/5
    - Key benefits: {', '.join(persona.pros) if isinstance(persona.pros, list) else persona.pros}
    - Issues encountered: {', '.join(persona.cons) if isinstance(persona.cons, list) else persona.cons}
    - Primary themes: {', '.join(persona.themes) if persona.themes else 'None specified'}
    - Recommendations: {', '.join(persona.suggestions) if persona.suggestions else 'None provided'}
    - Usage scenario: {persona.use_case}
    - Technical background: {persona.technical_level}

    Previous survey responses:
    """
    if persona.conversation_history:
        prompt += "The customer has previously responded as follows:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Given this context, simulate the probability distribution for their answer to: "{question}"

    Consider these factors:
    - Overall product satisfaction
    - Specific experiences with the product
    - Technical expertise level
    - Use case requirements and expectations

    List probabilities for each option below, ensuring they total 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Provide a response in the following JSON format:
    {
        "relevant": boolean (true if the question aligns with their experience, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "A clear explanation of how the reviewer's profile informed the distribution"
    }
    
    If the question doesn't align with the reviewer's experience, set "relevant": false and leave other fields blank.
    """

    schema = {
        "name": "product_reviewer_response_v2",
        "description": "Simulated product reviewer survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the customer's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_product_reviewer_prompt_v3(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Generate a probability-based survey response prediction prompt for product reviews."""
    prompt = f"""You are a simulation model for product review surveys. 
    Your role is to predict the probability distribution of responses a customer might give, considering their experience with the product and their technical background.

    Review profile:
    - Product info: {persona.product_name} ({persona.product_category})
    - Review summary: {persona.summary}
    - Experience level: {persona.technical_level}
    - Usage pattern: {persona.use_case}
    - Overall rating: {persona.rating}/5
    - Highlighted features: {', '.join(persona.pros) if isinstance(persona.pros, list) else persona.pros}
    - Reported issues: {', '.join(persona.cons) if isinstance(persona.cons, list) else persona.cons}
    - Key themes identified: {', '.join(persona.themes) if persona.themes else 'None specified'}

    Previous interactions:
    """
    if persona.conversation_history:
        prompt += "Context from earlier responses:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Based on the above, estimate the likelihood of their response to: "{question}"

    Consider:
    - Product satisfaction level
    - Technical expertise and usage context
    - Specific experiences mentioned
    - Overall sentiment patterns

    Specify probabilities for each option, ensuring the total equals 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Deliver your output as a JSON object with the following structure:
    {
        "relevant": boolean (true if the question applies to their experience, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "Explanation of the probability distribution based on the reviewer's profile"
    }
    
    If the question is irrelevant, set "relevant": false and leave other fields blank.
    """

    schema = {
        "name": "product_reviewer_response_v3",
        "description": "Simulated product reviewer survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the customer's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_product_reviewer_prompt_v4(persona: Persona, question: str, options: List[str]) -> Tuple[str, Dict]:
    """Build a comprehensive prompt for product review survey simulation"""
    prompt = f"""You are a survey response simulator for product reviews. 
    Your task is to generate realistic probability distributions for how a customer with this profile would respond, considering their product experience and technical background.

    Customer and product profile:
    - Product reviewed: {persona.product_name} - {persona.product_category}
    - Review title: {persona.title}
    - Overall rating: {persona.rating}/5
    - Product strengths: {', '.join(persona.pros) if isinstance(persona.pros, list) else persona.pros}
    - Product weaknesses: {', '.join(persona.cons) if isinstance(persona.cons, list) else persona.cons}
    - Technical proficiency: {persona.technical_level}
    - Main themes: {', '.join(persona.themes) if persona.themes else 'None specified'}
    - Suggested improvements: {', '.join(persona.suggestions) if persona.suggestions else 'None provided'}

    Previous response history:
    """
    if persona.conversation_history:
        prompt += "The reviewer has provided these previous responses:\n"
        for hist in persona.conversation_history:
            prompt += f"- {hist['summary']}\n"

    prompt += f"""
    Based on this profile, simulate the probability distribution for how this customer would respond to: "{question}"

    Consider:
    - Technical background and expertise level
    - Specific product experiences described
    - Overall satisfaction and rating given
    - Key themes and suggestions mentioned
    
    Provide probabilities for each option, ensuring they sum to 1:
    """
    for i, opt in enumerate(options, 1):
        prompt += f"        {i}. {opt}\n"

    prompt += """
    Return a JSON object with:
    {
        "relevant": boolean (true if question relates to their product experience, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "Detailed explanation of why this distribution makes sense for this reviewer"
    }
    
    If the question is not relevant to the reviewer's experience, set "relevant": false and leave other fields as empty strings.
    """

    schema = {
        "name": "product_reviewer_response_v4",
        "description": "Simulated product reviewer survey response with probability distribution using an array of option-probability objects.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "True if the question relates to the customer's experience, false otherwise."
                },
                "option": {
                    "type": "array",
                    "description": "An array of objects, each representing a response option and its probability.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "The text of the response option."
                            },
                            "probability": {
                                "type": "number",
                                "description": "The probability assigned to this option."
                            }
                        },
                        "required": ["option", "probability"],
                        "additionalProperties": False
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "A detailed explanation of how the probability distribution was determined."
                }
            },
            "required": ["relevant", "option", "reason"],
            "additionalProperties": False
        }
    }

    # Return the prompt and the schema
    return prompt, schema

def build_employee_personality_summary_prompt(persona: Persona) -> str:
    """Generate a prompt to summarize an employee's personality based on their profile."""
    
    prompt = """You are part of a team running simulations with personalities for survey analysis.
    You will be provided certain attributes of a persona/their responses or thoughts on something.
    You will be asked to summarize the personality of the persona based on the provided attributes.

    Your response should be a summary of the personality of the persona in 20 WORDS or less.
    """

    prompt += f"""
    The attributes are:
    - Name: {persona.name}
    - Role: {persona.role} in {persona.location}
    - Work experience: {persona.employment_status}
    - Rating of the company: {persona.rating}/5
    """

    if persona.pros:
        prompt += f"- Pros of your job: {persona.pros}\n"
    if persona.cons:
        prompt += f"- Cons of your job: {persona.cons}\n"

    sentiment = []
    if persona.recommend is not None and not persona.recommend:
        sentiment.append("Does not recommend")
    if persona.ceo_approval is not None and not persona.ceo_approval:
        sentiment.append("Does not approve of CEO") 
    if persona.business_outlook is not None and not persona.business_outlook:
        sentiment.append("Negative business outlook")
        
    if sentiment:
        prompt += f"- Overall sentiment: {', '.join(sentiment)}\n"

    if persona.advice_to_management:
        prompt += f"- Key concerns: {persona.advice_to_management}\n"

    return prompt

def build_product_reviewer_personality_summary_prompt(persona: Persona) -> str:
    """Generate a prompt to summarize a product reviewer's personality based on their review profile."""
    prompt = """You are part of a team running simulations with personalities for survey analysis.
    You will be provided certain attributes of a persona/their responses or thoughts on something.
    You will be asked to summarize the personality of the persona based on the provided attributes.

    Your response should be a summary of the personality of the persona in 20 WORDS or less.
    """

    prompt += f"""
    The attributes are:
    - Name: {persona.name}
    - Product reviewed: {persona.product_name} ({persona.product_category})
    - Location: {persona.location}
    - Rating given: {persona.rating}/5
    - Review title: {persona.title}
    """

    if persona.pros:
        pros = ', '.join(persona.pros) if isinstance(persona.pros, list) else persona.pros
        prompt += f"- Positive aspects: {pros}\n"
    
    if persona.cons:
        cons = ', '.join(persona.cons) if isinstance(persona.cons, list) else persona.cons
        prompt += f"- Negative aspects: {cons}\n"

    if persona.themes:
        themes = ', '.join(persona.themes)
        prompt += f"- Key themes: {themes}\n"

    sentiment = []
    if persona.recommend is not None and not persona.recommend:
        sentiment.append("Does not recommend product")
    if persona.technical_level:
        prompt += f"- Technical expertise: {persona.technical_level}\n"
    if persona.use_case:
        prompt += f"- Use case: {persona.use_case}\n"

    if sentiment:
        prompt += f"- Overall sentiment: {', '.join(sentiment)}\n"

    if persona.suggestions:
        suggestions = ', '.join(persona.suggestions)
        prompt += f"- Suggested improvements: {suggestions}\n"

    return prompt

