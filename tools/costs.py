import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("cost_log.jsonl")


class CostLogger:
    def log(
        self,
        *,
        provider: str,
        model: str,
        mode: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_s: float,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "mode": mode,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "latency_s": latency_s,
        }
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def dashboard(self) -> None:
        if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
            print("No cost data yet.")
            return

        entries = []
        for line in LOG_PATH.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))

        today = datetime.now(timezone.utc).date()
        week_start = today.toordinal() - today.weekday()

        total_all = 0.0
        total_today = 0.0
        total_week = 0.0
        by_provider: dict[str, float] = {}
        by_mode: dict[str, float] = {}
        priciest = entries[0]

        for e in entries:
            d = datetime.fromisoformat(e["timestamp"]).date()
            c = e["cost_usd"]
            total_all += c
            if d == today:
                total_today += c
            if d.toordinal() >= week_start:
                total_week += c
            by_provider[e["provider"]] = by_provider.get(e["provider"], 0.0) + c
            by_mode[e["mode"]] = by_mode.get(e["mode"], 0.0) + c
            if c > priciest["cost_usd"]:
                priciest = e

        print(f"Today: ${total_today:.6f}  |  This week: ${total_week:.6f}  |  All time: ${total_all:.6f}")

        print("\nBy provider:")
        for provider, cost in sorted(by_provider.items()):
            print(f"  {provider}: ${cost:.6f}")

        print("\nBy mode:")
        for mode, cost in sorted(by_mode.items()):
            print(f"  {mode}: ${cost:.6f}")

        p = priciest
        print(f"\nMost expensive: {p['provider']}/{p['model']} [{p['mode']}]")
        print(f"  Cost: ${p['cost_usd']:.6f}  Tokens: {p['prompt_tokens']} in / {p['completion_tokens']} out  Latency: {p['latency_s']:.2f}s")
