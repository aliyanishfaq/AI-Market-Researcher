import os
import typer
import tiktoken
from typing import Optional
from rich.console import Console
from dotenv import load_dotenv
from text_splitter import RecursiveCharacterTextSplitter

# Import AzureOpenAI
#from openai import AzureOpenAI
# Langfuse
from langfuse.openai import AzureOpenAI

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

def create_azure_openai_client(api_key: str, azure_endpoint: str) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=AZURE_ENDPOINT,
        api_version="2024-12-01-preview",
    )


def get_ai_client(service: str, console: Console) -> AzureOpenAI:
    # Decide which API key and endpoint to use
    if service.lower() == "azure":
        api_key = API_KEY
        endpoint = AZURE_ENDPOINT
        if not api_key or not endpoint:
            console.print("[red]Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment[/red]")
            raise typer.Exit(1)
        client = create_azure_openai_client(api_key=api_key, azure_endpoint=endpoint)

        return client
    else:
        console.print(
            "[red]Invalid service selected. Choose 'azure'.[/red]"
        )
        raise typer.Exit(1)


MIN_CHUNK_SIZE = 140
encoder = tiktoken.get_encoding(
    "cl100k_base"
)  # Updated to use OpenAI's current encoding

def trim_prompt(
    prompt: str, context_size: int = int(os.getenv("CONTEXT_SIZE", "128000"))
) -> str:
    """Trims a prompt to fit within the specified context size."""
    if not prompt:
        return ""

    length = len(encoder.encode(prompt))
    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    # Estimate characters to remove (3 chars per token on average)
    chunk_size = len(prompt) - overflow_tokens * 3
    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    trimmed_prompt = (
        splitter.split_text(prompt)[0] if splitter.split_text(prompt) else ""
    )

    # Handle edge case where trimmed prompt is same length
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)

    return trim_prompt(trimmed_prompt, context_size)
