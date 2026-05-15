"""Phase 8 demo (Bucket-1) — EverCore on its native data shape.

Three simulated AI ↔ user dialogues, each ~8-10 turns spanning a coherent
topic. POSTed as conversation messages with per-user session_ids.
NO flush call — let EverCore's LLM boundary detector decide when the
conversation has formed an episode worth extracting.

Compare to demo_two_agent_shared_knowledge.py (Bucket-2):
  Phase 4 demo: 1 fact per scroll → 2-turn synthetic wrap + forced flush
  Phase 8 demo: 8-10 turns of real dialogue per session → natural boundary

After all 3 dialogues:
  - GET episodic_memory per user → expect ≥1 episode each
  - SEARCH cross-user query → expect topic retrieval
  - GET profile per user → expect per-user profile facts populated

Pedagogical signal: EverCore extracts atomic_facts + builds profiles
WITHOUT the flush trickery from BCJ Entry 13. The pipeline EARNS its cost
when the data shape matches.

Run:
  python -m src.demo_conversational_imprint
"""
import asyncio
import json
import os
import time
import urllib.request

import httpx


EVERCORE = "http://localhost:1995"


# ── Three simulated dialogues (Bucket-1 shape) ─────────────────────────

DIALOGUES = [
    {
        "user_id": "alice",
        "session_id": "alice-api-rollout-2026-05",
        "topic": "Production API rollout planning",
        "turns": [
            ("user", "We're cutting a new API endpoint for the mobile team next week. What's the deploy plan?"),
            ("assistant", "For a new endpoint I'd run Terraform plan first against staging, then apply against prod after on-call sign-off. The standard module is modules/api-stack which already handles VPC peering + load balancer attachment."),
            ("user", "How long does the apply usually take?"),
            ("assistant", "First-deploy budget is 5 minutes wall-clock. Subsequent applies for config changes are under 2 minutes because the heavy resources are already provisioned."),
            ("user", "What if VPC peering isn't set up yet for the data-lake account?"),
            ("assistant", "Then you add a vpc-peering submodule call BEFORE the main api-stack module. The peering takes ~3 minutes to provision and the api-stack will block until it's active. Total first-deploy goes to ~8 minutes in that case."),
            ("user", "Do I need to update DNS too?"),
            ("assistant", "Yes — Route53 records are managed by a separate Terraform stack (dns-stack). After api-stack apply, run terraform apply against dns-stack with the new endpoint hostname. DNS propagation is usually under 60 seconds for in-VPC clients."),
            ("user", "Got it. I'll write a runbook for the team."),
            ("assistant", "Good call. Standard sequence: vpc-peering (if new account) → api-stack → dns-stack → smoke-test via /health endpoint. Document the 5-min apply budget so on-call doesn't ping you mid-rollout."),
        ],
    },
    {
        "user_id": "bob",
        "session_id": "bob-token-rotation-2026-05",
        "topic": "Auth token rotation",
        "turns": [
            ("user", "We had an incident yesterday — stale auth tokens caused 401s on the checkout service. How should rotation work?"),
            ("assistant", "Auth tokens should expire after 30 minutes by default. Stale tokens return HTTP 401 which the client SDK should catch and refresh from Keycloak. Was the SDK handling 401 correctly?"),
            ("user", "Apparently not — the client was caching tokens for an hour and never refreshed."),
            ("assistant", "That's the bug. Token TTL is enforced server-side but client-side caching has to respect Expires. Fix is to read the Keycloak token response's `expires_in` field and refresh at 80% of that — never cache past the server-asserted TTL."),
            ("user", "What's the right rotation cadence for the signing key itself?"),
            ("assistant", "Signing key rotation runs every 90 days via the keycloak-rotation cron job at midnight UTC. The job pushes the new key to all services + keeps the old key active for 24 hours so in-flight tokens stay valid through the transition."),
            ("user", "Where do I check that the rotation job actually ran?"),
            ("assistant", "Datadog dashboard `auth-rotation` shows last successful run timestamp. Alert fires if no successful rotation in 92 days. The job logs to CloudWatch under /aws/lambda/keycloak-rotation."),
            ("user", "Thanks. I'll fix the client SDK and add a test."),
            ("assistant", "Right approach. Test that the SDK refreshes at 80% TTL and that it falls back to a single retry on 401 from a token it thought was valid. Both bugs landed yesterday's incident."),
        ],
    },
    {
        "user_id": "carol",
        "session_id": "carol-incident-response-2026-05",
        "topic": "Incident response process",
        "turns": [
            ("user", "I'm joining the on-call rotation next month. Walk me through the incident response process?"),
            ("assistant", "On-call engineer triages within 5 minutes of page. Sev1 (customer-impacting) auto-pages secondary. Postmortem within 48 hours of resolution. Culture is blameless — root cause focus, not finger-pointing."),
            ("user", "How do I know if something is Sev1 vs Sev2?"),
            ("assistant", "Sev1 = revenue-impacting OR data-loss-risk OR security-incident. Sev2 = degraded service but no revenue impact. Sev3 = single-team or internal-tool issue. Datadog auto-classifies most pages but you can manually upgrade."),
            ("user", "What do I do in the first 15 minutes of a Sev1?"),
            ("assistant", "Acknowledge the page. Open the incident Slack channel via /incident. Post the symptom + your current hypothesis. Page secondary if you need help. Status page goes yellow within 5 minutes of page. Don't try to fix and communicate simultaneously — communicate first."),
            ("user", "Who writes the postmortem?"),
            ("assistant", "The incident commander (usually the on-call) writes the first draft within 48 hours. Template lives in Notion. Review meeting within a week with the affected team + SRE lead. Action items go into Linear with explicit owners + due dates."),
            ("user", "Are there metrics on MTTR I should be aware of?"),
            ("assistant", "Sev1 MTTR target is 60 minutes; current quarterly average is 47 minutes. Sev2 MTTR target 4 hours; we run ~2.5h. Both reported in the monthly engineering review. The dashboard is at grafana.internal/incidents."),
            ("user", "Got it. I'll read the runbook before my first shift."),
            ("assistant", "Good. Best prep: shadow the current on-call for one week. Watch how they handle the morning standup, the Datadog noise, the false-positive pages. Theory in the runbook + apprenticeship from real shifts is the actual training."),
        ],
    },
]


# ── Helpers ────────────────────────────────────────────────────────────


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{EVERCORE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req).read())


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── Imprint ────────────────────────────────────────────────────────────


def imprint_dialogue(dialogue: dict) -> tuple[float, dict, dict]:
    """POST a multi-turn dialogue + flush.

    Empirical finding (2026-05-15 lab run): even 10-12 turn natural
    dialogues return `accumulated` — EverCore's LLM boundary detector
    is GENUINELY conservative and won't fire on lab-scale conversations
    even when the data shape matches Bucket 1. Flush is still required
    at lab scale; the difference vs Phase 4 (Bucket 2) is the QUALITY
    of extracted atomic_facts + profile content, not the call sequence.
    """
    ts = _now_ms()
    messages = []
    for i, (role, content) in enumerate(dialogue["turns"]):
        messages.append({"role": role, "timestamp": ts + i, "content": content})

    body = {
        "user_id": dialogue["user_id"],
        "session_id": dialogue["session_id"],
        "messages": messages,
    }
    t0 = time.perf_counter()
    resp = _post("/api/v1/memories", body)
    flush_resp = _post(
        "/api/v1/memories/flush",
        {"user_id": dialogue["user_id"], "session_id": dialogue["session_id"]},
    )
    wall = time.perf_counter() - t0
    return wall, resp, flush_resp


# ── Verification probes ────────────────────────────────────────────────


def get_episodes(user_id: str) -> list[dict]:
    body = {"memory_type": "episodic_memory", "filters": {"user_id": user_id}, "page_size": 10}
    return _post("/api/v1/memories/get", body)["data"]["episodes"]


def get_profiles(user_id: str) -> list[dict]:
    body = {"memory_type": "profile", "filters": {"user_id": user_id}, "page_size": 10}
    return _post("/api/v1/memories/get", body).get("data", {}).get("profiles", [])


def search(query: str, user_id: str, k: int = 5) -> list[dict]:
    """Search scoped to ONE user_id at a time. EverCore's filters require
    at least user_id or group_id at the first level (empty filter → 422).
    For cross-user retrieval, call once per user and union the results."""
    body = {"query": query, "top_k": k, "filters": {"user_id": user_id}}
    return _post("/api/v1/memories/search", body).get("data", {}).get("episodes", [])


# ── Main ───────────────────────────────────────────────────────────────


async def main() -> None:
    print(">>> Phase 8 — Bucket-1 demo: 3 multi-turn dialogues, no flush\n")

    imprint_walls = []
    for d in DIALOGUES:
        wall, resp, flush_resp = imprint_dialogue(d)
        imprint_walls.append(wall)
        post_status = resp["data"].get("status", "?")
        flush_status = flush_resp["data"].get("status", "?")
        msg_count = resp["data"].get("message_count", "?")
        print(f"  {d['user_id']:8s} {d['session_id']:35s} {msg_count} turns "
              f"→ post={post_status} flush={flush_status}  wall={wall:.2f}s")

    print(f"\n  Mean imprint wall (incl. flush): {sum(imprint_walls)/len(imprint_walls):.2f}s")
    print("  Waiting 60s for EverCore async memcell extraction to complete...\n")
    time.sleep(60)

    # Verify episodes + profiles per user
    print(">>> Per-user verification\n")
    total_episodes = 0
    total_profiles = 0
    for d in DIALOGUES:
        eps = get_episodes(d["user_id"])
        profs = get_profiles(d["user_id"])
        total_episodes += len(eps)
        total_profiles += len(profs)
        print(f"  {d['user_id']:8s}  episodes={len(eps)}  profiles={len(profs)}")
        for e in eps[:1]:
            print(f"    └─ subject: {e.get('subject', '?')[:80]}")
            print(f"       summary: {(e.get('summary') or '?')[:120]}")

    # Cross-user topical search — query each user separately, union results.
    # EverCore's filters require user_id at first level (empty filter → 422).
    print("\n>>> Cross-user search (per-user, then union)\n")
    user_ids = [d["user_id"] for d in DIALOGUES]
    for q in [
        "how do we deploy production APIs",
        "what causes 401 errors",
        "what is the Sev1 MTTR target",
    ]:
        print(f"  Q: {q!r}")
        all_hits = []
        for uid in user_ids:
            for h in search(q, user_id=uid, k=2):
                h["__user_id"] = uid
                all_hits.append(h)
        all_hits.sort(key=lambda h: -(h.get("score") or 0.0))
        for h in all_hits[:3]:
            print(f"    score={h.get('score', 0):.3f}  user={h['__user_id']:8s}  "
                  f"subject={(h.get('subject') or '?')[:60]}")

    print("\n>>> Summary")
    print(f"  Dialogues imprinted: {len(DIALOGUES)}")
    print(f"  Total episodes extracted: {total_episodes}")
    print(f"  Total profiles built: {total_profiles}")
    print(f"  Mean per-dialogue imprint wall: {sum(imprint_walls)/len(imprint_walls):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
