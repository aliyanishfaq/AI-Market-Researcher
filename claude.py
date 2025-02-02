from anthropic import AsyncAnthropicBedrock
import asyncio
import json

claude_client = AsyncAnthropicBedrock(
    aws_region="us-west-2",
)

async def process_survey_response(prompt: str) -> str:
    message = await claude_client.messages.create(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=1024,
        tool_choice = {"type": "tool", "name": "process_survey_response"},
        temperature=0.0,
        tools=[{
        "name": "process_survey_response",
        "description": "Process the employee survey response prediction",
        "input_schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": "Whether the question is relevant to the employee's experience"
                },
                "option": {
                    "type": "object",
                    "description": "Probability distribution for each response option",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation for the probability distribution"
                }
            },
            "required": ["relevant", "option", "reason"]
            }
        }],
        messages=[{"role": "user", "content": """You are a simulation model for employee surveys. 
    Your role is to predict the probability distribution of responses an employee might give, considering the nuances of their profile and the potential for variation influenced by recent experiences or emotional states.

    Employee profile summary:
    - Position: Manufacturing engineer located in Chandler, AZ
    - Employment details: Former employee, more than 3 years
    - Positive job attributes: Benefits are great, lots of resources and impressive support structure to enable your success across the board. Many areas to explore if you aren't happy with your current role, and open to moving you if you need to. Workers seem cared for and taken care of in every aspect.
    - Negative job attributes: Ridiculous rules and rigidity. Millions of classes, signs everywhere warning you about holding the hand rail so you don't die. Depending on your department there are meetings every hour. Pressure comes from the chain as expected given Intel's performance in the market lately. Generally not a very entrepreneurial company, and you can feign innovation and creativity to get ahead even when you haven't done a lot. No one who makes decisions is spending their own money, so like most big companies, there ends up being a lot of bloat and lack of efficiency. Because of this, a lot of employees manufacture work so they stay relevant and no one wants to admit there are too many people doing the same job.
    - Overall company rating: 5.0/5
    - Sentiment analysis: Likely to recommend, Positive CEO approval, Positive business outlook
    - Primary concerns: Bring some small business and entrepreneurial spirit into the business, and move past the over-the-top rigidity and employee brainwashing attempts.

    Historical responses:
    
    Based on the above, estimate the likelihood of their response to the question: "What is the company's rating?"

    Take into account:
    - Daily mood shifts and relevant interactions
    - Patterns observed in their feedback and attitudes
    - Highlights and challenges within their job role
    - Their typical rating and sentiment trends

    Specify the probabilities for each option, ensuring the total equals 1:
    1. 1
    2. 2
    3. 3
    4. 4
    5. 5

    Deliver your output as a JSON object with the following structure:
    {
        "relevant": boolean (true if the question applies to the persona, false otherwise),
        "option": {option1: probability, option2: probability, ...},
        "reason": "Explanation of the assigned probabilities based on the persona's attributes"
    }
    
    If the question is irrelevant, set "relevant": false and leave the other fields blank."""}],
)

    tool_response = message.content[0].input
    #print(tool_response)  # Will contain {"relevant": bool, "option": {...}, "reason": "..."}
    return tool_response


async def main():
    response = await process_survey_response("What is the company's rating?")
    #validation for the distrbution needs to be added
    print(response)

    

if __name__ == "__main__":
    asyncio.run(main())