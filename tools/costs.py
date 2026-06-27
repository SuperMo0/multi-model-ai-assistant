import json
import os
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
        if not LOG_PATH.exists():
            print("No cost data yet — cost_log.jsonl not found.")
            return

        entries = []
        with LOG_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            print("No cost data yet.")
            return

        now = datetime.now(timezone.utc)
        today = now.date()
        week_start = today.toordinal() - today.weekday()

        total_all = 0.0
        total_today = 0.0
        total_week = 0.0
        by_provider: dict[str, float] = {}
        by_mode: dict[str, float] = {}
        most_expensive = entries[0]

        for entry in entries:
            ts = datetime.fromisoformat(entry["timestamp"]).date()
            cost = entry["cost_usd"]
            total_all += cost
            if ts == today:
                total_today += cost
            if ts.toordinal() >= week_start:
                total_week += cost
            by_provider[entry["provider"]] = by_provider.get(entry["provider"], 0.0) + cost
            by_mode[entry["mode"]] = by_mode.get(entry["mode"], 0.0) + cost
            if cost > most_expensive["cost_usd"]:
                most_expensive = entry

        width = os.get_terminal_size().columns

        def section(title: str) -> None:
            print(f"\n{'─' * width}")
            print(f"  {title}")
            print(f"{'─' * width}")

        section("COST DASHBOARD")
        print(f"  Today          ${total_today:.6f}")
        print(f"  This week      ${total_week:.6f}")
        print(f"  All time       ${total_all:.6f}")

        section("By provider")
        for provider, cost in sorted(by_provider.items()):
            print(f"  {provider:<20} ${cost:.6f}")

        section("By mode")
        for mode, cost in sorted(by_mode.items()):
            print(f"  {mode:<20} ${cost:.6f}")

        section("Most expensive call")
        e = most_expensive
        print(f"  {e['timestamp']}  {e['provider']} / {e['model']}")
        print(f"  Mode: {e['mode']}   Cost: ${e['cost_usd']:.6f}   Latency: {e['latency_s']:.2f}s")
        print(f"  Tokens: {e['prompt_tokens']} in / {e['completion_tokens']} out")
        print()
