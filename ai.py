import json
import time
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from llm import get_client
from tools.costs import CostLogger

app = typer.Typer(add_completion=False)
console = Console()
logger = CostLogger()
SESSIONS_DIR = Path("sessions")


class Provider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"


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


if __name__ == "__main__":
    app()
