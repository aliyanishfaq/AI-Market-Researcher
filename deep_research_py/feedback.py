from typing import List
import asyncio
import openai
import json
try:
    from prompt import system_prompt
except ImportError:
    from deep_research_py.prompt import system_prompt
from openai import AzureOpenAI

async def generate_feedback(query: str, client: AzureOpenAI, model: str) -> List[str]:
    """Generates follow-up questions to clarify research direction."""

    # Run OpenAI call in thread pool since it's synchronous
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {
                    "role": "user",
                    "content": f"Given this research topic: {query}, generate 3-5 follow-up questions to better understand the user's research needs. Return the response as a JSON object with a 'questions' array field.",
                },
            ],
            response_format={"type": "json_object"},
        ),
    )

    # Parse the JSON response
    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("questions", [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return []
