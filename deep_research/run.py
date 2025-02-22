import os
from dotenv import load_dotenv
import asyncio
import typer
from functools import wraps
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from openai import AzureOpenAI
from deep_research.deep_research import deep_research, write_final_report
from deep_research.feedback import generate_feedback
from deep_research.providers import get_ai_client

load_dotenv()

app = typer.Typer()
console = Console()
session = PromptSession()

async def async_prompt(message: str, default: str = "") -> str:
    """Async wrapper for prompt_toolkit."""
    return await session.prompt_async(message)

@app.command()
async def main(query: str, breadth: int = 4, depth: int = 2, concurrency: int = 2, 
                service: str = "azure", model: str = "o3-mini", quiet: bool = False):
    """Deep Research CLI"""
    print(f"[deep_research] [run] query: {query}")
    if not quiet:
        console.print(
            Panel.fit(
                "[bold blue]Deep Research Assistant[/bold blue]\n"
                "[dim]An AI-powered research tool[/dim]"
            )
        )
        console.print(f"🛠️ Using [bold green]{service.upper()}[/bold green] service.")

    client = get_ai_client(service, console)

    # Combine information
    combined_query = f"Initial Query: {query}"

    # Now use Progress for the research phase
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Do research
        task = progress.add_task(
            "[yellow]Researching your topic...[/yellow]", total=None
        )
        research_results = await deep_research(
            query=combined_query,
            breadth=breadth,
            depth=depth,
            concurrency=concurrency,
            client=client,
            model=model,
        )
        progress.remove_task(task)

        if not quiet:
            # Show learnings
            console.print("\n[yellow]Learnings:[/yellow]")
            for learning in research_results["learnings"]:
                rprint(f"• {learning}")

        # Generate report
        task = progress.add_task("Writing final report...", total=None)
        report = await write_final_report(
            prompt=combined_query,
            learnings=research_results["learnings"],
            visited_urls=research_results["visited_urls"],
            client=client,
            model=model
        )
        progress.remove_task(task)

        if not quiet:
            # Show results
            console.print("\n[bold green]Research Complete![/bold green]")
            console.print("\n[yellow]Final Report:[/yellow]")
            console.print(Panel(report, title="Research Report"))

            # Show sources
            console.print("\n[yellow]Sources:[/yellow]")
            for url in research_results["visited_urls"]:
                rprint(f"• {url}")

        # Save report with a filename based on the report title
        report_title = report.splitlines()[0].strip()  # Assuming the first line is the title
        safe_title = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in report_title)
        filename = f"{safe_title}.md"

        with open(filename, "w") as f:
            f.write(report)
        console.print(f"\n[dim]Report has been saved to {filename}[/dim]")

    return report

def run():
    """Synchronous entry point for the CLI tool."""
    typer.run(main)

if __name__ == "__main__":
    run()