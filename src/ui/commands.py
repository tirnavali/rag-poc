"""Slash-command dispatcher for the chat UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Action(Enum):
    EXIT = auto()
    CLEAR = auto()
    TOGGLE_DEBUG = auto()
    SHOW_HELP = auto()
    SHOW_SOURCE = auto()
    MUFETTIS = auto()
    RAPOR = auto()
    NORMAL_QUERY = auto()
    UNKNOWN_COMMAND = auto()
    EMPTY = auto()


@dataclass
class CommandResult:
    action: Action
    query: Optional[str] = None
    source_index: Optional[int] = None
    error_msg: Optional[str] = None


def parse_command(raw: str) -> CommandResult:
    """Parse a raw input line into a structured CommandResult."""
    if not raw.strip():
        return CommandResult(action=Action.EMPTY)

    cmd = raw.lower().strip()

    if cmd == "/cikis":
        return CommandResult(action=Action.EXIT)
    if cmd == "/temizle":
        return CommandResult(action=Action.CLEAR)
    if cmd == "/debug":
        return CommandResult(action=Action.TOGGLE_DEBUG)
    if cmd == "/yardim":
        return CommandResult(action=Action.SHOW_HELP)

    if cmd.startswith("/kaynak"):
        parts = cmd.split()
        if len(parts) < 2 or not parts[1].isdigit():
            return CommandResult(
                action=Action.SHOW_SOURCE,
                error_msg="Lütfen bir sayı girin. Örn: /kaynak 1",
            )
        return CommandResult(action=Action.SHOW_SOURCE, source_index=int(parts[1]) - 1)

    if cmd.startswith("/müfettiş") or cmd.startswith("/mufettis"):
        parts = raw.split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult(
                action=Action.MUFETTIS,
                error_msg="Lütfen bir soru girin. Örn: /müfettiş 1980 darbesi etkileri",
            )
        return CommandResult(action=Action.MUFETTIS, query=parts[1])

    if cmd.startswith("/rapor"):
        parts = raw.split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult(
                action=Action.RAPOR,
                error_msg="Lütfen bir konu girin. Örn: /rapor 1997 Refah Partisi kapatma davası",
            )
        return CommandResult(action=Action.RAPOR, query=parts[1])

    if cmd.startswith("/"):
        return CommandResult(action=Action.UNKNOWN_COMMAND, error_msg=f"Bilinmeyen komut: {cmd}")

    return CommandResult(action=Action.NORMAL_QUERY, query=raw)
