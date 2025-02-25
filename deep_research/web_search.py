from typing import List, Dict, TypedDict, Optional
import asyncio
import os
from firecrawl import FirecrawlApp
import inspect
from dotenv import load_dotenv

load_dotenv()

class SearchResponse(TypedDict):
    data: List[Dict[str, str]]

FIRECRAWL_API_KEY=os.getenv("FIRECRAWL_API_KEY")

class Firecrawl:
    """Simple wrapper for Firecrawl SDK."""

    def __init__(self, api_url: Optional[str] = None):
        self.app = FirecrawlApp(api_key=FIRECRAWL_API_KEY, api_url=api_url)

    async def search(self, query: str, timeout: int = 15000, limit: int = 5) -> SearchResponse:
        """Search using Firecrawl SDK in a thread pool to keep it async."""
        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.app.search(
                    query=query,
                    params={
                        "limit": 5,
                        "scrapeOptions": {
                            "formats": ["markdown"]
                        }
                    }
                )
            )
            # Handle the response format from the SDK
            if isinstance(response, dict) and "data" in response:
                print("[deep_research] [Firecrawl] Handling dict with data")
                print("[deep_research] [Firecrawl] Response", response)
                return response
            elif isinstance(response, dict) and "success" in response:
                print("[deep_research] [Firecrawl] Handling dict with success")
                return {"data": response.get("data", [])}
            elif isinstance(response, list):
                print("[deep_research] [Firecrawl] Handling list response")
                # Response is a list of results
                formatted_data = []
                for item in response:
                    if isinstance(item, dict):
                        formatted_data.append(item)
                    else:
                        # Handle non-dict items (like objects)
                        formatted_data.append(
                            {
                                "url": getattr(item, "url", ""),
                                "markdown": getattr(item, "markdown", "")
                                or getattr(item, "content", ""),
                                "title": getattr(item, "title", "")
                                or getattr(item, "metadata", {}).get("title", ""),
                            }
                        )
                return {"data": formatted_data}
            else:
                print(f"Unexpected response format from Firecrawl: {type(response)}")
                return {"data": []}

        except Exception as e:
            print(f"Error searching with Firecrawl: {e}")
            print(
                f"Response type: {type(response) if 'response' in locals() else 'N/A'}"
            )
            return {"data": []}
