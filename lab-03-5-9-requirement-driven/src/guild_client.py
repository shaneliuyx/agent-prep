"""Re-export of the shared MCP-stdio guild client.

Canonical implementation lives once in agent-prep/shared/guild_client.py (single
source of truth for the W3.5.x lab cluster: W3.5.5 guild, W3.5.8 two-tier,
W3.5.9 requirement-driven). This shim preserves the established
`from src.guild_client import ...` path used across the labs without re-vendoring
the 153-LOC implementation into every lab.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))  # repo-root/shared on path
from guild_client import *  # noqa: E402,F401,F403
from guild_client import GuildClient, is_accept_winner, QuestStatus  # noqa: E402,F401
