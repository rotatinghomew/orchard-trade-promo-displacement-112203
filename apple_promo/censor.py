"""售罄之后的缺单不是零需求。

``listings.sold_out`` 之后观察不到完整购买机会，直接拿当天销量当需求会把
被截断的部分算成 0。做法：

1. 用同一商品、同一排期基准（轮换 / 日历）、未售罄窗口的分时销量比例，
   估计「售罄时点前应占全天的份额」；
2. 该窗口的观测销量除以这个份额，得到包含缺单的全天需求；
3. 缺口 = 全天需求 − 观测销量。

已知小时是整点至下一整点，和 :10 / :00 的售罄边界可能落在同一小时桶，
因此给出两种口径并取范围：剔除重叠小时桶（保守）与保留（宽松）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .packet import Packet


@dataclass(frozen=True)
class CensorAdjustment:
    window: str
    sku: str
    zone: str
    sold_out: str
    observed: int
    share_before: float
    inferred_full_day: float
    add_back: float
    basis: str


@dataclass(frozen=True)
class CensorReport:
    adjustments: tuple[CensorAdjustment, ...]
    add_back_low: float
    add_back_high: float

    def by_window(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for adj in self.adjustments:
            result[adj.window] = result.get(adj.window, 0.0) + adj.add_back
        return result

    def total(self) -> float:
        return 0.5 * (self.add_back_low + self.add_back_high)


def _profile(
    packet: Packet,
    sku: str,
    basis: str,
    as_of: str | None,
) -> dict[int, float]:
    """未售罄窗口的分时销量份额。"""
    counts: dict[int, float] = {}
    for listing in packet.listings:
        if listing.sku != sku or listing.sold_out is not None:
            continue
        rollout = packet.rollout_by_window[listing.window]
        if rollout.basis != basis:
            continue
        for order in packet.orders_by_window.get(listing.window, []):
            if order.original != sku:
                continue
            counts[order.known_hour] = counts.get(order.known_hour, 0.0) + packet.net_units(
                order, as_of
            )
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {hour: value / total for hour, value in counts.items()}


def estimate_censoring(
    packet: Packet,
    as_of: str | None = None,
    *,
    basis: str = "rotation",
) -> CensorReport:
    adjustments: list[CensorAdjustment] = []
    for listing in packet.sellouts():
        rollout = packet.rollout_by_window[listing.window]
        if rollout.basis != basis:
            continue
        cutoff_hour = listing.sold_out_hour
        observed = sum(
            packet.net_units(order, as_of)
            for order in packet.orders_by_window.get(listing.window, [])
            if order.original == listing.sku
        )
        base = _profile(packet, listing.sku, basis, as_of)
        # 保守：售罄所在整点桶无法拆分，只计更早的整点；
        # 宽松：该整点桶整体计入售罄前。份额越小，补回越多。
        share_low = sum(v for h, v in base.items() if h < cutoff_hour)
        share_high = sum(v for h, v in base.items() if h <= cutoff_hour)
        if share_low <= 0:
            inferred = float(observed)
            add_back = 0.0
            share = 0.0
        else:
            inferred = observed / share_low
            add_back = max(0.0, inferred - observed)
            share = share_low
        adjustments.append(
            CensorAdjustment(
                window=listing.window,
                sku=listing.sku,
                zone=listing.zone,
                sold_out=listing.sold_out or "",
                observed=observed,
                share_before=share,
                inferred_full_day=inferred,
                add_back=add_back,
                basis=basis,
            )
        )
    return CensorReport(
        tuple(adjustments),
        add_back_low=sum(a.add_back for a in adjustments),
        add_back_high=sum(a.add_back for a in adjustments),
    )


def window_units_with_censoring(
    packet: Packet,
    as_of: str | None = None,
    *,
    basis: str = "rotation",
    use_censoring: bool = True,
) -> dict[str, int]:
    """窗口净销量，售罄窗口按保守份额补回缺单（四舍五入到整件）。"""
    units = {
        window: packet.window_net_units(window, as_of) for window in packet.rollout_by_window
    }
    if not use_censoring or basis != "rotation":
        return units
    report = estimate_censoring(packet, as_of, basis=basis)
    for adj in report.adjustments:
        # 用较保守的补回量（观测 ÷ 放宽份额），避免高估。
        if adj.share_before > 0:
            extra = adj.observed / adj.share_before - adj.observed
        else:
            extra = 0.0
        units[adj.window] = units.get(adj.window, 0) + int(round(extra))
    return units
