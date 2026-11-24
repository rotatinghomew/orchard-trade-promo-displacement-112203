"""资料包的读取、引用核对与统一口径。

本模块把原始记录整理成只读对象，并集中定义所有分析共用的口径：

* 净销量：退款（含 ``assess_as_of`` 之前已知的更正）整笔冲销，不计销量；
* 券数：以核销编号为准，商品行数不能反推券数；
* 库存：盘点数已含早期退货，只有盘点后被验收的可再售商品才能加回；
* 会员保留量：到 ``member_release`` 当日开门前仍不可动用。

任何口径调整都只在此处做一次，避免各分析各算一套。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

SKUS = ("apple", "dried", "juice")

#: 四种投放状态，``none`` 表示这一轮确认没有投放。
STATES = ("none", "C1", "C3", "C1+C3")


class PacketError(ValueError):
    """资料不自洽时抛出。"""


def _as_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def state_label(offered) -> str:
    """把 ``rollouts.offered``（券种列表）转成状态标签。"""
    kinds = sorted(offered)
    if not kinds:
        return "none"
    return "+".join(kinds)


@dataclass(frozen=True)
class Policy:
    min_net_margin: int
    min_comparable_blocks: int
    pilot_limit: int
    assess_as_of: str
    zone_groups: dict[str, str]
    stock_counted_at: str

    @property
    def assess_date(self) -> date:
        return _as_date(self.assess_as_of)


@dataclass(frozen=True)
class Listing:
    window: str
    zone: str
    week: int
    day: int
    sku: str
    promo: bool
    seen: int | None
    sold_out: str | None
    members_set_aside: int

    @property
    def sold_out_minute(self) -> int | None:
        if self.sold_out is None:
            return None
        hour, minute = self.sold_out.split(":")
        return int(hour) * 60 + int(minute)

    @property
    def sold_out_hour(self) -> int | None:
        if self.sold_out is None:
            return None
        return int(self.sold_out.split(":")[0])


@dataclass(frozen=True)
class Order:
    id: str
    window: str
    zone: str
    household: str
    original: str
    actual: str
    qty: int
    coupon: str | None
    member: bool
    refund: bool
    resellable: bool
    known_hour: int

    @property
    def substituted(self) -> bool:
        return self.original != self.actual


@dataclass(frozen=True)
class Rollout:
    window: str
    block: str
    eligible: tuple[str, ...]
    offered: tuple[str, ...]
    basis: str

    @property
    def state(self) -> str:
        return state_label(self.offered)


@dataclass(frozen=True)
class Exposure:
    window: str
    household: str
    promo: str
    campaign: str
    zone: str
    cohort: str


@dataclass(frozen=True)
class Bundle:
    id: str
    price: int
    components: dict[str, int]
    alt: dict[str, list[str]]

    @property
    def nominal_units(self) -> int:
        return sum(self.components.values())


@dataclass(frozen=True)
class StockCell:
    store: str
    sku: str
    base: int
    reserved: int
    unit_cost: int
    member_release: str


@dataclass(frozen=True)
class Plan:
    id: str
    bundle: str
    store: str
    limit: int
    status: str

    @property
    def issued(self) -> bool:
        return self.status == "issued"


@dataclass(frozen=True)
class Update:
    order: str
    refund: bool
    resellable: bool
    known: str


@dataclass(frozen=True)
class Redemption:
    id: str
    window: str
    household: str
    coupon: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class StockReceipt:
    id: str
    order: str
    store: str
    sku: str
    qty: int
    received: str
    accepted: bool


class Packet:
    """只读资料包。"""

    def __init__(self, raw: dict[str, Any], source: str | None = None) -> None:
        self.source = source
        self.raw = raw
        self.policy = Policy(**raw["policy"])
        self.listings = [Listing(**row) for row in raw["listings"]]
        self.orders = [Order(**row) for row in raw["orders"]]
        self.exposures = [Exposure(**row) for row in raw["exposures"]]
        self.bundles = [Bundle(**row) for row in raw["bundle"]]
        self.stock = [StockCell(**row) for row in raw["stock"]]
        self.plans = [Plan(**row) for row in raw["planned"]]
        self.updates = [Update(**row) for row in raw["updates"]]
        self.rollouts = [
            Rollout(**{**row, "eligible": tuple(row["eligible"]), "offered": tuple(row["offered"])})
            for row in raw["rollouts"]
        ]
        self.redemptions = [
            Redemption(**{**row, "lines": tuple(row["lines"])}) for row in raw["redemptions"]
        ]
        self.receipts = [StockReceipt(**row) for row in raw["stock_receipts"]]

        self.orders_by_id = {o.id: o for o in self.orders}
        self.orders_by_window: dict[str, list[Order]] = {}
        for order in self.orders:
            self.orders_by_window.setdefault(order.window, []).append(order)
        self.listings_by_window: dict[str, list[Listing]] = {}
        for listing in self.listings:
            self.listings_by_window.setdefault(listing.window, []).append(listing)
        self.rollout_by_window = {r.window: r for r in self.rollouts}
        self.exposures_by_window: dict[str, list[Exposure]] = {}
        for exposure in self.exposures:
            self.exposures_by_window.setdefault(exposure.window, []).append(exposure)
        self.bundle_by_id = {b.id: b for b in self.bundles}
        self.stock_by_key = {(s.store, s.sku): s for s in self.stock}
        self.plan_by_id = {p.id: p for p in self.plans}
        self.updates_by_order = {u.order: u for u in self.updates}
        self.redemptions_by_window: dict[str, list[Redemption]] = {}
        for redemption in self.redemptions:
            self.redemptions_by_window.setdefault(redemption.window, []).append(redemption)
        self.receipts_by_order = {r.order: r for r in self.receipts}

        self.validate()

    # ------------------------------------------------------------------ 读取
    @classmethod
    def load(cls, path: str | Path) -> "Packet":
        path = Path(path)
        try:
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except OSError as exc:  # pragma: no cover - 由调用方处理
            raise PacketError(f"无法读取资料：{exc}") from exc
        if not isinstance(raw, dict) or any(
            not isinstance(value, (list, dict)) for value in raw.values()
        ):
            raise PacketError("资料包必须是由记录列表或参数对象组成的 JSON 对象")
        return cls(raw, source=str(path))

    # ------------------------------------------------------------------ 核对
    def validate(self) -> None:
        windows = {row.window for row in self.listings}
        if {row.window for row in self.orders} - windows:
            raise PacketError("成交记录引用了不存在的窗口")
        if {row.window for row in self.exposures} - windows:
            raise PacketError("曝光记录引用了不存在的窗口")
        if {row.window for row in self.rollouts} != windows:
            raise PacketError("排期记录与营业记录窗口不一致")
        if len(self.orders_by_id) != len(self.orders):
            raise PacketError("成交编号重复")
        if {row.order for row in self.updates} - set(self.orders_by_id):
            raise PacketError("退款更正引用了不存在的成交")
        if {row.bundle for row in self.plans} - set(self.bundle_by_id):
            raise PacketError("投放计划引用了不存在的券")
        for order in self.orders:
            if order.resellable and not order.refund:
                raise PacketError(f"{order.id} 未退款却标记可再售")
            if order.zone != order.window.split("-")[1]:
                raise PacketError(f"{order.id} 门店与窗口不一致")
        for redemption in self.redemptions:
            rollout = self.rollout_by_window[redemption.window]
            if redemption.coupon not in rollout.offered:
                raise PacketError(f"{redemption.id} 的券种不在该窗口投放")
            for line in redemption.lines:
                order = self.orders_by_id.get(line)
                if order is None:
                    raise PacketError(f"{redemption.id} 引用了不存在的商品行")
                if (order.coupon, order.household, order.window) != (
                    redemption.coupon,
                    redemption.household,
                    redemption.window,
                ):
                    raise PacketError(f"{redemption.id} 的商品行归属不一致")
        for update in self.updates:
            if update.resellable and not update.refund:
                raise PacketError(f"{update.order} 未退款却标记可再售")
        for receipt in self.receipts:
            order = self.orders_by_id[receipt.order]
            update = self.updates_by_order.get(receipt.order)
            if update is None or receipt.accepted != update.resellable:
                raise PacketError(f"{receipt.id} 的验收与退款更正不一致")
            if receipt.sku != order.actual:
                raise PacketError(f"{receipt.id} 的商品与成交净交付不一致")
            if _as_datetime(receipt.received) <= _as_datetime(self.policy.stock_counted_at):
                raise PacketError(f"{receipt.id} 早于盘点时点，不能加回库存")

    # ------------------------------------------------------------------ 口径
    def effective_refund(self, order: Order, as_of: str | None = None) -> bool:
        """评估时点已知的退款状态。

        ``updates`` 只改变退款状态，不产生新成交；晚于评估时点的更正不采用。
        """
        cutoff = _as_date(as_of or self.policy.assess_as_of)
        update = self.updates_by_order.get(order.id)
        if update is not None and _as_date(update.known) <= cutoff:
            return update.refund
        return order.refund

    def net_units(self, order: Order, as_of: str | None = None) -> int:
        return 0 if self.effective_refund(order, as_of) else order.qty

    def window_net_units(self, window: str, as_of: str | None = None) -> int:
        return sum(self.net_units(o, as_of) for o in self.orders_by_window.get(window, []))

    def window_state(self, window: str) -> str:
        return self.rollout_by_window[window].state

    def zone_group(self, zone: str) -> str:
        return self.policy.zone_groups[zone]

    def zone_group_members(self, group: str) -> list[str]:
        return sorted(z for z, g in self.policy.zone_groups.items() if g == group)

    def bucket_key(self, order: Order) -> str:
        """互斥且完备的分项：会员/非会员 × 券/非券。"""
        return ("member" if order.member else "nonmember") + (
            "_coupon" if order.coupon else "_plain"
        )

    def window_buckets(self, window: str, as_of: str | None = None) -> dict[str, int]:
        """按分项汇总净销量，并附上总量、券量、会员量与替换量。"""
        buckets = {
            "member_coupon": 0,
            "member_plain": 0,
            "nonmember_coupon": 0,
            "nonmember_plain": 0,
        }
        totals = {"total": 0, "coupon": 0, "member": 0, "substituted": 0}
        for order in self.orders_by_window.get(window, []):
            units = self.net_units(order, as_of)
            buckets[self.bucket_key(order)] += units
            totals["total"] += units
            if order.coupon:
                totals["coupon"] += units
            if order.member:
                totals["member"] += units
            if order.substituted:
                totals["substituted"] += units
        result = {**buckets, **totals}
        assert sum(buckets.values()) == totals["total"]
        return result

    def coupon_bounds(self, window: str) -> tuple[int, int]:
        """该窗口券数的可行范围。

        有统一核销编号时是精确值；早期收银流水只有商品行上的券种，
        不能由行数反推券数，只给上下界：至少每户每券一张，至多每行一张。
        """
        redemptions = self.redemptions_by_window.get(window)
        if redemptions:
            return len(redemptions), len(redemptions)
        groups: set[tuple[str, str]] = set()
        lines = 0
        for order in self.orders_by_window.get(window, []):
            if order.coupon:
                groups.add((order.household, order.coupon))
                lines += 1
        if not lines:
            return 0, 0
        return len(groups), lines

    def coupon_count(self, window: str) -> int | None:
        """精确券数；早期缺少核销编号时返回 ``None``。"""
        redemptions = self.redemptions_by_window.get(window)
        return len(redemptions) if redemptions else None

    # ------------------------------------------------------------------ 库存
    def member_reserved(self, store: str, sku: str, as_of: str | None = None) -> int:
        """到点前仍被会员保留、不可动用的数量。"""
        cell = self.stock_by_key[(store, sku)]
        cutoff = _as_date(as_of or self.policy.assess_as_of)
        return cell.reserved if cutoff < _as_date(cell.member_release) else 0

    def issued_committed(self, store: str, sku: str) -> int:
        """已发权益占用的组件量：核销与否都占最大用量。"""
        total = 0
        for plan in self.plans:
            if not plan.issued or plan.store != store:
                continue
            bundle = self.bundle_by_id[plan.bundle]
            total += bundle.components.get(sku, 0) * plan.limit
        return total

    def accepted_returns(self, store: str, sku: str) -> int:
        """盘点后仓管实收、当前可再售的商品量。"""
        counted = _as_datetime(self.policy.stock_counted_at)
        return sum(
            receipt.qty
            for receipt in self.receipts
            if receipt.store == store
            and receipt.sku == sku
            and receipt.accepted
            and _as_datetime(receipt.received) > counted
        )

    def gross_available_units(self, store: str, sku: str, as_of: str | None = None) -> int:
        """可动用的实物量 = 盘点数 − 会员保留量 + 盘点后验收入库。

        这一口径**不**扣已发权益：已发权益只是优先占用，实物仍在仓里，
        要放进分配队列里按优先级领取，否则会出现「先扣一次、再领一次」
        的重复计算。
        """
        cell = self.stock_by_key[(store, sku)]
        return cell.base - self.member_reserved(store, sku, as_of) + self.accepted_returns(store, sku)

    def available_units(self, store: str, sku: str, as_of: str | None = None) -> int:
        """未被已发权益占用的可售量，用于回答「还能不能再加新券」。"""
        return self.gross_available_units(store, sku, as_of) - self.issued_committed(store, sku)

    # ------------------------------------------------------------------ 其他
    def digest(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:16]

    def sellouts(self) -> list[Listing]:
        return [row for row in self.listings if row.sold_out is not None]

    def missing_seen(self) -> list[Listing]:
        return [row for row in self.listings if row.seen is None]

    def households_in_week(self, week: int) -> dict[str, set[str]]:
        """同一周内某户出现过的门店，用于识别跨店购买机会。"""
        result: dict[str, set[str]] = {}
        for order in self.orders:
            if int(order.window.split("-")[0]) != week:
                continue
            result.setdefault(order.household, set()).add(order.zone)
        return result
