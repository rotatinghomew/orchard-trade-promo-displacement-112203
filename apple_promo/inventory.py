"""库存按真实组件扣减、组合共享商品与等量替换。

关键口径：

* 可用量 = 盘点数 − 会员保留量 − 已发权益占用 + 盘点后验收入库；
* 已发权益（``status=issued``）无论是否核销都先占最大用量，重跑分析不能让它失效；
* 组合共享商品时，先满足无法替换的需求，再让可替换的组合争用余量；
* ``alt`` 只允许逐项等量替换，不改变售价；
* 每张可兑现券扣除真实组件成本后的净收益必须达到 ``min_net_margin``，
  替换后也要守住，因此按「最不利替换」核算收益。
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

from .packet import Packet, SKUS


@dataclass
class Allocation:
    store: str
    bundle: str
    requested: int
    allocated: int
    nominal: dict[str, int] = field(default_factory=dict)
    substitutions: dict[str, dict[str, int]] = field(default_factory=dict)
    shortfall: dict[str, int] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not any(self.shortfall.values())


@dataclass(frozen=True)
class MarginCheck:
    bundle: str
    price: int
    nominal_cost: int
    nominal_margin: int
    worst_cost: int
    worst_margin: int
    floor: int

    @property
    def passes(self) -> bool:
        return self.worst_margin >= self.floor


@dataclass
class StoreState:
    store: str
    physical: dict[str, int]   # 盘点 − 会员保留 + 实收入库
    budget: dict[str, int]     # physical − 已发权益占用，可用于新券
    reserved: dict[str, int]
    committed: dict[str, int]
    received: dict[str, int]


def store_state(packet: Packet, store: str, as_of: str | None = None) -> StoreState:
    physical, budget, reserved, committed, received = {}, {}, {}, {}, {}
    for sku in SKUS:
        physical[sku] = packet.gross_available_units(store, sku, as_of)
        committed[sku] = packet.issued_committed(store, sku)
        budget[sku] = physical[sku] - committed[sku]
        reserved[sku] = packet.member_reserved(store, sku, as_of)
        received[sku] = packet.accepted_returns(store, sku)
    return StoreState(store, physical, budget, reserved, committed, received)


def margin_check(packet: Packet, bundle_id: str) -> MarginCheck:
    bundle = packet.bundle_by_id[bundle_id]
    costs = {sku: packet.stock_by_key[("E", sku)].unit_cost for sku in SKUS}
    nominal = sum(costs[sku] * qty for sku, qty in bundle.components.items())
    worst = nominal
    for src, alts in bundle.alt.items():
        for alt in alts:
            delta = (costs[alt] - costs[src]) * bundle.components.get(src, 0)
            worst = max(worst, nominal + delta)
    return MarginCheck(
        bundle=bundle_id,
        price=bundle.price,
        nominal_cost=nominal,
        nominal_margin=bundle.price - nominal,
        worst_cost=worst,
        worst_margin=bundle.price - worst,
        floor=packet.policy.min_net_margin,
    )


def _trial(bundle, count: int, free: dict[str, int]) -> tuple[dict, dict, dict]:
    """试算 ``count`` 张能否兑现；不修改 ``free``。"""
    remaining = dict(free)
    nominal: dict[str, int] = {}
    substitutions: dict[str, dict[str, int]] = {}
    short: dict[str, int] = {}
    for sku, per in bundle.components.items():
        need = per * count
        take = min(need, max(0, remaining.get(sku, 0)))
        remaining[sku] = remaining.get(sku, 0) - take
        nominal[sku] = take
        if take < need:
            short[sku] = need - take
    for src, alts in bundle.alt.items():
        deficit = short.get(src, 0)
        if not deficit:
            continue
        for alt in alts:
            take = min(deficit, max(0, remaining.get(alt, 0)))
            if take:
                remaining[alt] = remaining.get(alt, 0) - take
                substitutions.setdefault(src, {})[alt] = take
                deficit -= take
            if deficit == 0:
                break
        if deficit:
            short[src] = deficit
        else:
            short.pop(src, None)
    return nominal, substitutions, short


def allocate(
    packet: Packet,
    store: str,
    requests: list[tuple[str, int]],
    as_of: str | None = None,
) -> tuple[list[Allocation], dict[str, int]]:
    """按给定优先顺序分配组件；返回每项实际分配与剩余可用量。

    ``requests`` 的顺序就是优先级：已发权益排前面，试算的组合排后面。
    每项只按「真正能兑现的张数」扣减，未兑现的部分不占库存，因此不会
    因为前一项申请过大而把后面的组合全部挤掉。
    """
    state = store_state(packet, store, as_of)
    free = dict(state.physical)
    results: list[Allocation] = []
    for bundle_id, count in requests:
        bundle = packet.bundle_by_id[bundle_id]
        feasible_count = 0
        chosen = None
        for candidate in range(count, -1, -1):
            nominal, substitutions, short = _trial(bundle, candidate, free)
            if not short:
                feasible_count = candidate
                chosen = (nominal, substitutions)
                break
        if chosen is None:
            chosen = ({sku: 0 for sku in bundle.components}, {})
        nominal, substitutions = chosen
        # 落账：只扣实际使用的组件。
        for sku, qty in nominal.items():
            free[sku] = free.get(sku, 0) - qty
        for src, mapping in substitutions.items():
            for alt, qty in mapping.items():
                free[alt] = free.get(alt, 0) - qty
        shortfall = {}
        if feasible_count < count:
            # 报告相对申请的缺口，便于解释「为什么只能批这么多」。
            trial_nominal, trial_sub, trial_short = _trial(bundle, count, dict(state.physical))
            shortfall = trial_short or {"*": count - feasible_count}
        results.append(
            Allocation(
                store=store,
                bundle=bundle_id,
                requested=count,
                allocated=feasible_count,
                nominal=nominal,
                substitutions=substitutions,
                shortfall=shortfall,
            )
        )
    return results, free


def max_feasible(packet: Packet, store: str, bundle_id: str, as_of: str | None = None) -> int:
    """在不挤压其它组合余量时，单个组合最多能兑现多少张。

    可替换组件与替代品共享余量，所以先求一个松弛上界：对每个组件，
    把自身与可用替代品的余量合起来除以其单张用量，取所有约束的最小值。
    这个上界一定不小于真实最大张数，再在其上二分收敛。
    """
    state = store_state(packet, store, as_of)
    bundle = packet.bundle_by_id[bundle_id]
    upper = sum(state.physical.values())
    for sku, per in bundle.components.items():
        if not per:
            continue
        pool = state.physical.get(sku, 0) + sum(
            state.physical.get(alt, 0) for alt in bundle.alt.get(sku, [])
        )
        upper = min(upper, pool // per)
    low, high = 0, upper
    while low < high:
        mid = (low + high + 1) // 2
        if not _trial(bundle, mid, state.physical)[2]:
            low = mid
        else:
            high = mid - 1
    return low


def component_plan(
    packet: Packet,
    allocations: list[Allocation],
) -> dict[str, dict[str, int]]:
    """把分配结果折算成各门店各 SKU 的实际扣减（含替代品）。"""
    plan: dict[str, dict[str, int]] = {}
    for alloc in allocations:
        store = plan.setdefault(alloc.store, {sku: 0 for sku in SKUS})
        for sku, qty in alloc.nominal.items():
            store[sku] = store.get(sku, 0) + qty
        for src, mapping in alloc.substitutions.items():
            for alt, qty in mapping.items():
                store[alt] = store.get(alt, 0) + qty
    return plan


def shared_component_pressure(
    packet: Packet,
    allocations: list[Allocation],
) -> dict[str, dict[str, float]]:
    """同一门店里，多个组合对同一件商品的争用程度。

    返回每店每 SKU 的「申请量 / 可用量」，大于 1 说明在抢同一件商品。
    """
    result: dict[str, dict[str, float]] = {}
    by_store: dict[str, list[Allocation]] = collections.defaultdict(list)
    for alloc in allocations:
        by_store[alloc.store].append(alloc)
    for store, allocs in by_store.items():
        state = store_state(packet, store)
        demand: dict[str, int] = collections.Counter()
        for alloc in allocs:
            for sku, qty in alloc.nominal.items():
                demand[sku] += qty
            for src, mapping in alloc.substitutions.items():
                for alt, qty in mapping.items():
                    demand[alt] += qty
        result[store] = {
            sku: (demand.get(sku, 0) / state.physical[sku])
            if state.physical.get(sku)
            else float("inf")
            for sku in SKUS
        }
    return result
