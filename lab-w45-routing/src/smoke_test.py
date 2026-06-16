"""Probe every fleet endpoint with a 1-token completion + record idle latency."""
import os
import time

from openai import OpenAI

from src.fleet_config import FLEET


def probe(name: str, ep) -> tuple[bool, float, str]:
    client = OpenAI(base_url=ep.base_url, api_key=os.getenv("OMLX_API_KEY"))
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=ep.model,
            messages=[{"role": "user", "content": "Reply with one token: ok"}],
            max_tokens=4,
            temperature=0.0,
        )
        wall = time.perf_counter() - t0
        return True, wall, (r.choices[0].message.content or "").strip()
    except Exception as e:                                             # noqa: BLE001
        return False, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    for name, ep in FLEET.items():
        ok, wall, reply = probe(name, ep)
        status = "OK" if ok else "FAIL"
        print(f"{status:4s}  {name:11s} {ep.model:50s} {wall*1000:6.0f}ms  reply={reply!r}")