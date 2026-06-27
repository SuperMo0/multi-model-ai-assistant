import asyncio
import json
import time
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel

from llm import get_client
from llm.openai_client import CAPABLE_MODEL as OAI_CAPABLE, DEFAULT_MODEL as OAI_DEFAULT
from llm.anthropic_client import CAPABLE_MODEL as ANT_CAPABLE, DEFAULT_MODEL as ANT_DEFAULT
from tools.costs import CostLogger
from tools.document import read_file, DocumentSummary

app = typer.Typer(add_completion=False)
console = Console()
logger = CostLogger()
SESSIONS_DIR = Path("sessions")


class Provider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"


class Detail(str, Enum):
    brief = "brief"
    standard = "standard"
    detailed = "detailed"


DETAIL_INSTRUCTIONS = {
    Detail.brief: "Provide a brief summary with only the most important points.",
    Detail.standard: "Provide a standard summary covering all key topics.",
    Detail.detailed: "Provide a thorough and detailed summary of the entire document.",
}


def load_session(name: str) -> list[dict]:
    path = SESSIONS_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_session(name: str, history: list[dict]) -> None:
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{name}.json"
    path.write_text(json.dumps(history, indent=2))


def log_response(result: dict, provider: str, mode: str, latency: float) -> None:
    logger.log(
        provider=provider,
        model=result["model"],
        mode=mode,
        prompt_tokens=result["tokens"]["prompt"],
        completion_tokens=result["tokens"]["completion"],
        cost_usd=result["cost"],
        latency_s=latency,
    )


@app.command()
def chat(
    provider: Annotated[Provider, typer.Option("--provider", "-p")] = Provider.openai,
    model: Annotated[str, typer.Option("--model", "-m")] = "",
    session: Annotated[str, typer.Option("--session", "-s")] = "default",
    system: Annotated[str, typer.Option("--system")] = "You are a helpful assistant.",
    stream: Annotated[bool, typer.Option("--stream")] = False,
) -> None:
    client = get_client(provider.value)
    if model:
        client.model = model

    history = load_session(session)
    console.print(f"[dim]Session: {session}  Provider: {provider.value}  Model: {client.model}[/dim]\n")

    while True:
        try:
            user_input = typer.prompt(typer.style("you", fg=typer.colors.CYAN))
        except (KeyboardInterrupt, EOFError):
            save_session(session, history)
            console.print("\n[dim]Session saved.[/dim]")
            break

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": system}] + history[-20:]

        start = time.perf_counter()
        result = client.complete(messages, stream=stream)
        latency = time.perf_counter() - start

        if not result:
            console.print("[red]Request failed.[/red]")
            history.pop()
            continue

        if not stream:
            console.print(f"[green]{result['text']}[/green]")

        history.append({"role": "assistant", "content": result["text"]})
        save_session(session, history)
        log_response(result, provider.value, "chat", latency)

        tokens = result["tokens"]
        console.print(
            f"[dim]tokens: {tokens['prompt']}+{tokens['completion']}  "
            f"cost: ${result['cost']:.6f}  latency: {latency:.2f}s[/dim]\n"
        )


@app.command()
def analyse(
    file: Annotated[str, typer.Argument()],
    provider: Annotated[Provider, typer.Option("--provider", "-p")] = Provider.openai,
    detail: Annotated[Detail, typer.Option("--detail")] = Detail.standard,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        text, word_count = read_file(file)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    capable = ANT_CAPABLE if provider == Provider.anthropic else OAI_CAPABLE
    default = ANT_DEFAULT if provider == Provider.anthropic else OAI_DEFAULT
    model = capable if word_count > 10_000 else default

    client = get_client(provider.value)

    schema = DocumentSummary.model_json_schema()
    prompt = (
        f"{DETAIL_INSTRUCTIONS[detail]}\n\n"
        f"Respond with a JSON object matching this schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Document:\n{text}"
    )

    messages = [
        {"role": "system", "content": "You are a document analysis assistant. Respond only with valid JSON."},
        {"role": "user", "content": prompt},
    ]

    start = time.perf_counter()
    result = client.complete(messages, model=model)
    latency = time.perf_counter() - start

    if not result:
        console.print("[red]Request failed.[/red]")
        raise typer.Exit(1)

    try:
        raw = result["text"].strip().removeprefix("```json").removesuffix("```").strip()
        summary = DocumentSummary.model_validate_json(raw)
    except Exception as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise typer.Exit(1)

    log_response(result, provider.value, "analyse", latency)

    if as_json:
        print(summary.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]{summary.title}[/bold]")
    console.print(f"Words: {summary.word_count}  |  Sentiment: {summary.sentiment}\n")
    console.print("[bold]Topics:[/bold]")
    for t in summary.main_topics:
        console.print(f"  • {t}")
    console.print("\n[bold]Key points:[/bold]")
    for p in summary.key_points:
        console.print(f"  • {p}")
    if summary.recommended_actions:
        console.print("\n[bold]Recommended actions:[/bold]")
        for a in summary.recommended_actions:
            console.print(f"  • {a}")
    console.print(f"\n[dim]cost: ${result['cost']:.6f}  latency: {latency:.2f}s[/dim]")


async def _call(provider: str, prompt: str) -> tuple[str, dict, float]:
    client = get_client(provider)
    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    result = await asyncio.get_event_loop().run_in_executor(None, lambda: client.complete(messages))
    latency = time.perf_counter() - start
    return provider, result, latency


@app.command()
def compare(
    prompt: Annotated[str, typer.Argument()],
) -> None:
    async def run():
        return await asyncio.gather(
            _call("openai", prompt),
            _call("anthropic", prompt),
        )

    results = asyncio.run(run())

    panels = []
    for provider, result, latency in results:
        if not result:
            panels.append(Panel("[red]Failed[/red]", title=provider))
            continue
        t = result["tokens"]
        footer = f"tokens: {t['prompt']}+{t['completion']}  cost: ${result['cost']:.6f}  latency: {latency:.2f}s"
        panels.append(Panel(result["text"], title=provider, subtitle=footer))
        log_response(result, provider, "compare", latency)

    console.print(Columns(panels, equal=True))


@app.command()
def costs() -> None:
    logger.dashboard()


@app.command()
def costs() -> None:
    logger.dashboard()


if __name__ == "__main__":
    app()
