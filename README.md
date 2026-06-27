# Multi-model AI assistant

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-%23412991.svg?style=for-the-badge&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-%23CC785C.svg?style=for-the-badge&logo=anthropic&logoColor=white)
![Typer](https://img.shields.io/badge/Typer-000000?style=for-the-badge&logo=fastapi&logoColor=white)
![Rich](https://img.shields.io/badge/Rich-%23FAB005.svg?style=for-the-badge&logo=python&logoColor=black)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=python&logoColor=white)

![demo](assets/preview.gif)

this project is part of my AI learning journey.

in this project I built a CLI tool to chat interactively with OpenAI and Anthropic. I used the builder pattern with an abstract base class that all AI providers need to implement, this way I can switch between models easily without affecting the business code.

I used Typer to read CLI arguments and commands.

we have 4 commands

1. chat — start an interactive chat session.
2. analyse — analyse a .txt or .pdf file and print a structured summary.
3. compare — send the same prompt to both providers and show responses side by side.
4. costs — show cost breakdown from logged API calls.

flags

each command has its own flags but there are some shared ones:

`--provider` / `-p` — pick `openai` or `anthropic`, defaults to openai  
`--model` / `-m` — override the default model if you want a specific one  
`--session` / `-s` — name for the session, defaults to `default`, session history is saved to a json file so you can pick up where you left off  
`--system` — custom system prompt, defaults to "You are a helpful assistant."  
`--stream` — stream the response token by token instead of waiting for the full reply  

the `analyse` command has a couple extra ones:

`--detail` — how thorough the summary should be, options are `brief`, `standard`, or `detailed`, defaults to standard  
`--json` — dump the raw json output instead of the pretty printed version  

---

## setup

you need python and uv installed

```bash
git clone <repo>
cd multi-model-ai-assistant
uv sync
```

then create a `.env` file in the root with your api keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## how to run

```bash
# interactive chat
uv run ai.py chat

# chat with anthropic and a named session
uv run ai.py chat -p anthropic -s my-session

# analyse a file
uv run ai.py analyse report.pdf

# analyse with more detail and use anthropic
uv run ai.py analyse notes.txt -p anthropic --detail detailed

# compare both providers on the same prompt
uv run ai.py compare "explain how neural networks work"

# see your cost breakdown
uv run ai.py costs
```

---

## how it works

the core idea is a simple abstract base class in `llm/base.py` that defines what any AI provider needs to implement — basically just a `complete()` method. then `openai_client.py` and `anthropic_client.py` each implement that.

this means the command code in `ai.py` doesn't care which provider you picked, it just calls `client.complete()` and gets back an `LLMResponse` object with the text, token counts, and cost. adding a new provider later would just mean writing one new file.

every request gets logged to `cost_log.jsonl` so the `costs` command can show you a breakdown of what you've spent across providers and commands.

the `compare` command runs both providers in parallel using asyncio so you're not waiting for them one after the other.

---

## project structure

```text
ai.py              # cli entry point, all 4 commands live here
llm/
  base.py          # abstract base class for providers
  openai_client.py
  anthropic_client.py
tools/
  costs.py         # logging and cost dashboard
  document.py      # pdf/txt reading and summary schema
sessions/          # saved chat histories (json files)
cost_log.jsonl     # append-only log of every api call
```
