import json
from pathlib import Path
import json
from collections import defaultdict
from datetime import datetime, timedelta, UTC

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
            "timestamp": datetime.now(UTC).isoformat(),
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

        entries = [
            json.loads(line)
            for line in LOG_PATH.read_text().splitlines()
            if line.strip()
        ]

        today = datetime.now(UTC).date()
        week_start = today - timedelta(days=today.weekday())

        total_all = 0.0
        total_today = 0.0
        total_week = 0.0

        by_provider = defaultdict(float)
        by_mode = defaultdict(float)

        for e in entries:
            d = datetime.fromisoformat(e["timestamp"]).date()
            c = e["cost_usd"]

            total_all += c
            if d == today:
                total_today += c
            if d >= week_start:
                total_week += c

            by_provider[e["provider"]] += c
            by_mode[e["mode"]] += c

        print(f"Today: ${total_today:.6f}  |  This week: ${total_week:.6f}  |  All time: ${total_all:.6f}")

        print("\nBy provider:")
        for provider, cost in sorted(by_provider.items()):
            print(f"  {provider}: ${cost:.6f}")

        print("\nBy mode:")
        for mode, cost in sorted(by_mode.items()):
            print(f"  {mode}: ${cost:.6f}")

        p = max(entries, key=lambda x: x["cost_usd"])
        print(f"\nMost expensive: {p['provider']}/{p['model']} [{p['mode']}]")
        print(f"  Cost: ${p['cost_usd']:.6f}  Tokens: {p['prompt_tokens']} in / {p['completion_tokens']} out  Latency: {p['latency_s']:.2f}s")
