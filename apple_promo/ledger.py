"""批准时点快照与恢复。

经办人需要能在项目内恢复任一批准时点的方案。快照记录：

* 当时的资料指纹（``packet.digest()``），保证同一份输入；
* 被批准的行（门店、组合、张数、已发 / 待发）；
* 计算出的增量、范围与库存占用，便于回看「哪段预估代替了原来的消费」。

恢复语义：

* 快照不可变，只追加；
* 恢复某个时点只返回当时可见的方案，不用后来的资料重算历史结论；
* 若当前资料与快照指纹不同，明确标注「已过期，仅供回看」，不静默替换。
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SnapshotLine:
    plan_id: str
    bundle: str
    store: str
    limit: int
    status: str
    locked: bool
    allocated: int
    net_increment: float
    increment_low: float
    increment_high: float


@dataclass(frozen=True)
class Snapshot:
    label: str
    created_at: str
    packet_digest: str
    assess_as_of: str
    lines: tuple[SnapshotLine, ...]
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "created_at": self.created_at,
            "packet_digest": self.packet_digest,
            "assess_as_of": self.assess_as_of,
            "lines": [asdict(line) for line in self.lines],
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Snapshot":
        return cls(
            label=raw["label"],
            created_at=raw["created_at"],
            packet_digest=raw["packet_digest"],
            assess_as_of=raw["assess_as_of"],
            lines=tuple(SnapshotLine(**line) for line in raw["lines"]),
            context=raw.get("context", {}),
        )


class Ledger:
    """追加式快照账本，落盘为单个 JSON 文件。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: list[Snapshot] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        self._entries = [Snapshot.from_dict(item) for item in raw.get("snapshots", [])]

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"snapshots": [entry.to_dict() for entry in self._entries]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def add(self, snapshot: Snapshot) -> None:
        if any(entry.label == snapshot.label for entry in self._entries):
            raise ValueError(f"快照标签重复：{snapshot.label}")
        self._entries.append(snapshot)
        self._flush()

    def labels(self) -> list[str]:
        return [entry.label for entry in self._entries]

    def latest(self) -> Snapshot | None:
        return self._entries[-1] if self._entries else None

    def restore(self, label: str, current_digest: str | None = None) -> dict[str, Any]:
        """恢复某一批准时点的方案；不修改快照本身。"""
        for entry in self._entries:
            if entry.label == label:
                stale = current_digest is not None and entry.packet_digest != current_digest
                return {
                    "snapshot": copy.deepcopy(entry),
                    "stale": stale,
                    "note": (
                        "当前资料已与快照不同，仅作历史回看，不能据此后算"
                        if stale
                        else "资料指纹一致，可据此后算"
                    ),
                }
        raise KeyError(f"找不到快照：{label}")

    def diff(self, before: str, after: str) -> dict[str, Any]:
        """比较两个时点，标出哪段预估被替换、哪些组合仍在抢共享商品。"""
        first = self.restore(before)["snapshot"]
        second = self.restore(after)["snapshot"]
        bmap = {line.plan_id: line for line in first.lines}
        amap = {line.plan_id: line for line in second.lines}
        changed = []
        for plan_id in sorted(set(bmap) | set(amap)):
            old = bmap.get(plan_id)
            new = amap.get(plan_id)
            if old is None:
                changed.append(
                    {"plan_id": plan_id, "change": "新增", "to": asdict(new)}
                )
            elif new is None:
                changed.append(
                    {"plan_id": plan_id, "change": "移除", "from": asdict(old)}
                )
            elif asdict(old) != asdict(new):
                changed.append(
                    {
                        "plan_id": plan_id,
                        "change": "调整",
                        "from": asdict(old),
                        "to": asdict(new),
                        "delta_limit": new.limit - old.limit,
                        "delta_increment": new.net_increment - old.net_increment,
                    }
                )
        return {
            "before": before,
            "after": after,
            "changed": changed,
            "shared_component_notes": second.context.get("shared_components", {}),
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
