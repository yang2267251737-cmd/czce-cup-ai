"""品种池：从 ``config/universe.json`` 读取候选品种，并接到行情与合约规格上。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "config" / "universe.json"


@dataclass(frozen=True)
class UniverseEntry:
    """一个候选品种在池子里的登记信息。"""

    code: str
    name: str
    continuous: str
    reason: str
    tier: str


def load_universe(path: Path | None = None, tiers: tuple[str, ...] = ("core",)) -> list[UniverseEntry]:
    """读取品种池；``tiers`` 决定取哪些层级（core / satellite）。"""
    payload = json.loads((path or UNIVERSE_PATH).read_text(encoding="utf-8"))
    entries: list[UniverseEntry] = []
    for tier in tiers:
        for item in payload.get(tier, []):
            entries.append(
                UniverseEntry(
                    code=item["code"],
                    name=item.get("name", item["code"]),
                    continuous=item["continuous"],
                    reason=item.get("reason", ""),
                    tier=tier,
                )
            )
    return entries


def excluded_reasons(path: Path | None = None) -> dict[str, str]:
    """读取被排除品种及原因，方便在报告里"讲清楚为什么不选它们"。"""
    payload = json.loads((path or UNIVERSE_PATH).read_text(encoding="utf-8"))
    return {item["code"]: item["reason"] for item in payload.get("excluded", [])}


def symbols(entries: list[UniverseEntry]) -> list[str]:
    """取连续合约代码列表。"""
    return [entry.continuous for entry in entries]


def code_of(continuous: str) -> str:
    """把 ``CF0`` 还原成品种代码 ``CF``。"""
    return continuous.rstrip("0123456789") or continuous
