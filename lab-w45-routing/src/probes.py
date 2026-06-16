"""Load + split + validate the hand-labelled probe set."""
import json
import random
from pathlib import Path
from typing import Literal

Tier = Literal["haiku", "sonnet", "opus"]
Mode = Literal["minimal", "react", "deliberate"]


def load_probes(path: str = "tests/router_probes.jsonl") -> list[dict]:
    """Load + validate every row has the expected fields + values."""
    rows = []
    valid_tiers = {"haiku", "sonnet", "opus"}
    valid_modes = {"minimal", "react", "deliberate"}
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        for k in ("prompt", "expected_tier", "expected_mode", "domain"):
            if k not in row:
                raise ValueError(f"row {i}: missing field {k!r}")
        if row["expected_tier"] not in valid_tiers:
            raise ValueError(f"row {i}: bad tier {row['expected_tier']!r}")
        if row["expected_mode"] not in valid_modes:
            raise ValueError(f"row {i}: bad mode {row['expected_mode']!r}")
        rows.append(row)
    return rows


def train_eval_split(rows: list[dict], seed: int = 42, train_frac: float = 0.67):
    """Stratified split: roughly preserve tier+mode distribution in both halves."""
    rng = random.Random(seed)
    by_label: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["expected_tier"], r["expected_mode"])
        by_label.setdefault(key, []).append(r)
    train, eval_ = [], []
    for key, group in by_label.items():
        rng.shuffle(group)
        cut = int(len(group) * train_frac)
        train.extend(group[:cut])
        eval_.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_