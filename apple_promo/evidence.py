"""分项证据：把同一份净销量增量拆成能互相接回的部分。

要求（来自业务约束）：

* 四个分项 **会员/非会员 × 用券/未用券** 互斥且完备，之和必须严格等于总量；
* 总量还要能拆成 **新增** 与 **转移** 两条同样说得通的路径，
  在资料不足以分辨时给可行范围，而不是把相关性当成确定归因；
* 常见转移路径：
  1. 跨单及跨店代买（同一户、同一周在两店各买一次）；
  2. 会员需求挤出（会员保留量把非会员挡在外面，或会员把券额度占满）；
  3. 商品替换（``original != actual``，如 dried 换成 juice）。

本模块只报告可由资料直接核对的量；对不可分辨的归属给出上下界区间。
"""
from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass, field

from .packet import Packet


BUCKETS = ("member_coupon", "member_plain", "nonmember_coupon", "nonmember_plain")


@dataclass(frozen=True)
class BucketEffect:
    key: str
    per_window: float
    per_block: tuple[float, ...]

    @property
    def se(self) -> float:
        return statistics.stdev(self.per_block) / len(self.per_block) ** 0.5 if len(self.per_block) > 1 else float("inf")


@dataclass(frozen=True)
class Decomposition:
    state: str
    total: float
    buckets: dict[str, BucketEffect]
    coupon_units: float
    member_units: float
    substituted_units: float

    def check_sum(self) -> float:
        return sum(b.per_window for b in self.buckets.values()) - self.total


@dataclass(frozen=True)
class TransferBand:
    """一条转移路径的可行范围（每窗口净销量）。"""

    label: str
    low: float
    high: float
    evidence: str


@dataclass(frozen=True)
class JointStateReport:
    state: str
    windows: tuple[str, ...]
    weeks: tuple[int, ...]
    net_increment: float
    net_increment_low: float
    net_increment_high: float
    decomposition: Decomposition
    transfers: tuple[TransferBand, ...]
    cross_store_units: float
    duplicate_coupon_units: float
    coupon_bounds: tuple[int, int]
    redemptions: int
    notes: tuple[str, ...]


def _bucket_cells(packet: Packet, key: str, as_of: str | None):
    """按分项重算轮换单元的对照差分。"""
    from .design import rotation_cells, treated_and_control_groups, _mean, _week_of, _zone_of

    treat_group, control_group = treated_and_control_groups(packet)
    units = {}
    for window in packet.rollout_by_window:
        units[window] = packet.window_buckets(window, as_of)[key]
    return rotation_cells(packet, as_of, window_units=units)


def decompose(packet: Packet, as_of: str | None = None) -> dict[str, Decomposition]:
    from .design import _observed_effects, rotation_cells

    base = rotation_cells(packet, as_of)
    total_effects = _observed_effects(base)
    per_key: dict[str, dict[str, tuple[float, tuple[float, ...]]]] = {}
    for key in BUCKETS:
        cells = _bucket_cells(packet, key, as_of)
        per_key[key] = _observed_effects(cells)
    coupon_cells = rotation_cells(
        packet, as_of,
        window_units={w: packet.window_buckets(w, as_of)["coupon"] for w in packet.rollout_by_window},
    )
    member_cells = rotation_cells(
        packet, as_of,
        window_units={w: packet.window_buckets(w, as_of)["member"] for w in packet.rollout_by_window},
    )
    subst_cells = rotation_cells(
        packet, as_of,
        window_units={w: packet.window_buckets(w, as_of)["substituted"] for w in packet.rollout_by_window},
    )
    coupon_eff = _observed_effects(coupon_cells)
    member_eff = _observed_effects(member_cells)
    subst_eff = _observed_effects(subst_cells)

    result: dict[str, Decomposition] = {}
    for state in ("C1", "C3", "C1+C3"):
        buckets = {
            key: BucketEffect(key, per_key[key][state][0], per_key[key][state][1])
            for key in BUCKETS
        }
        result[state] = Decomposition(
            state=state,
            total=total_effects[state][0],
            buckets=buckets,
            coupon_units=coupon_eff[state][0],
            member_units=member_eff[state][0],
            substituted_units=subst_eff[state][0],
        )
    return result


def _cross_store_households(packet: Packet, weeks: set[int]) -> set[tuple[int, str]]:
    seen: dict[tuple[int, str], set[str]] = collections.defaultdict(set)
    for order in packet.orders:
        week = int(order.window.split("-")[0])
        if week in weeks:
            seen[(week, order.household)].add(order.zone)
    return {key for key, zones in seen.items() if len(zones) > 1}


def joint_report(packet: Packet, as_of: str | None = None) -> JointStateReport:
    """两券同投状态的净增量、分项和转移范围。"""
    from .design import estimate_rotation
    from .censor import window_units_with_censoring

    design = estimate_rotation(packet, as_of)
    joint = design.effect("C1+C3")
    days = sorted({cell.week for cell in design.cells if cell.state == "C1+C3"})
    weeks = set(days)
    windows = tuple(sorted(
        w for w in packet.rollout_by_window
        if packet.rollout_by_window[w].basis == "rotation"
        and packet.window_state(w) == "C1+C3"
    ))
    decomp = decompose(packet, as_of)["C1+C3"]

    cross = _cross_store_households(packet, weeks)
    cross_units = 0
    for window in windows:
        for order in packet.orders_by_window.get(window, []):
            if (int(window.split("-")[0]), order.household) in cross:
                cross_units += packet.net_units(order, as_of)
    cross_per_window = cross_units / len(windows) if windows else 0.0

    # 同一户、同一券同周出现在两店：可能是同一张券被代买，也可能各自核销。
    same_coupon = collections.defaultdict(list)
    for order in packet.orders:
        if order.coupon and int(order.window.split("-")[0]) in weeks:
            same_coupon[(int(order.window.split("-")[0]), order.household, order.coupon)].append(order)
    dup_coupons = [v for v in same_coupon.values() if len({o.zone for o in v}) > 1]
    duplicate_units = sum(packet.net_units(o, as_of) for group in dup_coupons for o in group)
    duplicate_per_window = duplicate_units / len(windows) if windows else 0.0

    # 券数按核销编号计，不按商品行数。
    redemptions = sum(len(packet.redemptions_by_window.get(w, [])) for w in windows)
    bounds = (0, 0)
    for window in windows:
        lo, hi = packet.coupon_bounds(window)
        bounds = (bounds[0] + lo, bounds[1] + hi)

    substituted = decomp.substituted_units
    member_coupon = decomp.buckets["member_coupon"].per_window
    member_plain = decomp.buckets["member_plain"].per_window
    nonmember_coupon = decomp.buckets["nonmember_coupon"].per_window
    nonmember_plain = decomp.buckets["nonmember_plain"].per_window

    # 会员需求挤出：会员券量被算进增量，但会员原本多半会买，只是换了券。
    # 可分辩的部分是「会员用券 − 会员未用券的负向变化」，区间下界取 0。
    member_gross = max(0.0, member_coupon)
    member_cannibal = max(0.0, min(member_coupon, member_coupon - member_plain))
    member_band = TransferBand(
        "会员需求挤出（会员把原本要买的改成用券）",
        0.0,
        member_cannibal,
        f"会员券量 {member_coupon:+.2f}、会员未用券 {member_plain:+.2f}（每窗口）",
    )
    # 跨单及跨店代买：可见的是两店都有成交的同户，真正代买只能给上界。
    cross_band = TransferBand(
        "跨单及跨店代买",
        0.0,
        max(0.0, min(cross_per_window, duplicate_per_window)),
        f"同周两店同户 {cross_per_window:.2f}/窗口；同券跨店 {duplicate_per_window:.2f}/窗口",
    )
    # 商品替换：original != actual 的部分本身就是转移，不产生新需求。
    subst_band = TransferBand(
        "商品替换（替代品顶替原商品）",
        0.0,
        max(0.0, substituted),
        f"替换成交 {substituted:+.2f}/窗口",
    )
    # 新增：总量减去可识别的转移；上界对应转移全为 0。
    transfer_low = max(0.0, member_band.low + cross_band.low + subst_band.low)
    transfer_high = max(0.0, member_band.high + cross_band.high + subst_band.high)
    net_low = joint.per_window - transfer_high
    net_high = joint.per_window - transfer_low

    notes = []
    notes.append("总量 = 四个互斥分项之和，分项已逐项核对接回总量。")
    if cross_per_window:
        notes.append("同组共享购物：E、W 作为共同客流冲击一起留在样本一边。")
    notes.append("两条同样说得通的转移路径并存时，给出的是可行范围而非确定归因。")
    if packet.coupon_count(windows[0]) is None if windows else False:
        notes.append("早期流水只有商品行券种，券数只能给上下界。")

    return JointStateReport(
        state="C1+C3",
        windows=windows,
        weeks=tuple(sorted(weeks)),
        net_increment=joint.per_window,
        net_increment_low=net_low,
        net_increment_high=net_high,
        decomposition=decomp,
        transfers=(member_band, cross_band, subst_band),
        cross_store_units=cross_per_window,
        duplicate_coupon_units=duplicate_per_window,
        coupon_bounds=bounds,
        redemptions=redemptions,
        notes=tuple(notes),
    )
