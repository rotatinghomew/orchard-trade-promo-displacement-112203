"""新组合试算：同时投放的新增与挤出范围、库存与收益约束、可缩减与不可动部分。

处理三类问题：

1. **两组同时投放**：各自的增量不能简单相加，因为可能把同一批顾客和商品
   算成各自增量。用轮换数据里的「两券同投」单元相对「单券」之和的差额，
   给出重叠（挤出）范围。
2. **可预见的共同替换情形**：两个组合共享同一件可替换商品时（例如都允许
   dried→juice），要守住会员保留量和最低活动收益要求。
3. **缩量**：新的退货或曝光更正使原计划要缩减时，必须区分
   * 可缩减的后续投放；
   * 必须继续留给已经领券或会员的部分（已发权益不能因重跑分析失效）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .packet import Packet, STATES
from .inventory import (
    Allocation,
    allocate,
    component_plan,
    margin_check,
    max_feasible,
    shared_component_pressure,
    store_state,
)


@dataclass(frozen=True)
class OverlapRange:
    """两组同时投放相对各自单投之和的挤出范围（每窗口净销量）。"""

    bundle_a: str
    bundle_b: str
    joint_effect: float
    single_a: float
    single_b: float

    @property
    def additive_sum(self) -> float:
        return self.single_a + self.single_b

    @property
    def overlap(self) -> float:
        """重复计入的部分；正值表示同投比各自单投之和少。"""
        return self.additive_sum - self.joint_effect

    @property
    def low(self) -> float:
        return min(self.joint_effect, self.additive_sum)

    @property
    def high(self) -> float:
        return max(self.joint_effect, self.additive_sum)


@dataclass(frozen=True)
class PlanLine:
    plan_id: str
    bundle: str
    store: str
    requested: int
    allocated: int
    status: str
    locked: bool
    margin_ok: bool
    notes: tuple[str, ...] = ()


@dataclass
class TrialResult:
    store_results: dict[str, list[PlanLine]]
    allocations: list[Allocation]
    component_plan: dict[str, dict[str, int]]
    pressure: dict[str, dict[str, float]]
    overlap: OverlapRange | None
    warnings: list[str]
    revenue_floor_ok: bool
    reserved_intact: bool


def overlap_from_design(packet: Packet, as_of: str | None = None, *, bundle_a="C1", bundle_b="C3") -> OverlapRange | None:
    """用轮换拉丁方估计的同投效应与单投之和比较。"""
    from .design import estimate_rotation

    design = estimate_rotation(packet, as_of)
    try:
        joint = design.effect(f"{bundle_a}+{bundle_b}")
    except KeyError:
        return None
    single_a = design.effect(bundle_a)
    single_b = design.effect(bundle_b)
    return OverlapRange(
        bundle_a=bundle_a,
        bundle_b=bundle_b,
        joint_effect=joint.per_window,
        single_a=single_a.per_window,
        single_b=single_b.per_window,
    )


def _issued_first(packet: Packet, store: str) -> list[tuple[str, int]]:
    """已发权益排在试算之前，保证重跑分析不会让已发券失效。"""
    issued = []
    for plan in packet.plans:
        if plan.store == store and plan.issued:
            issued.append((plan.bundle, plan.limit))
    return issued


def trial(
    packet: Packet,
    proposals: list[tuple[str, str, int]],
    *,
    as_of: str | None = None,
    apply_shrink: dict[str, int] | None = None,
) -> TrialResult:
    """试算若干 (计划编号, 组合编号, 张数)。

    ``apply_shrink`` 给出因新证据要缩减的张数；只缩减后续投放，
    已发权益部分保持不动。
    """
    apply_shrink = apply_shrink or {}
    by_store: dict[str, list[tuple[str, int, str, bool]]] = {}
    supplied_ids: set[str] = set()
    for plan_id, bundle_id, count in proposals:
        plan = packet.plan_by_id.get(plan_id)
        store = plan.store if plan else _default_store(packet, bundle_id)
        locked = bool(plan and plan.issued)
        supplied_ids.add(plan_id)
        by_store.setdefault(store, []).append((bundle_id, count, plan_id, locked))

    allocations: list[Allocation] = []
    lines: dict[str, list[PlanLine]] = {}
    warnings: list[str] = []
    # 未显式传入的已发权益也要参与排队占位（它们的实物占用必须先扣）。
    reported_stores = set(by_store)  # 有试算行的门店
    placeholder_plans = [
        plan for plan in packet.plans if plan.issued and plan.id not in supplied_ids
    ]
    for plan in placeholder_plans:
        by_store.setdefault(plan.store, [])
    for store, items in by_store.items():
        placeholders = [
            (p.bundle, p.limit, p.id, True) for p in placeholder_plans if p.store == store
        ]
        explicit_issued = [it for it in items if it[3]]
        explicit_proposed = [it for it in items if not it[3]]
        ordered = placeholders + explicit_issued + explicit_proposed
        requests = [(bundle, count) for bundle, count, _, _ in ordered]
        allocs, free = allocate(packet, store, requests, as_of)
        if store in reported_stores:
            lines[store] = []
        placeholder_ids = {p.id for p in placeholder_plans}
        for (bundle_id, count, plan_id, locked), alloc in zip(ordered, allocs):
            check = margin_check(packet, bundle_id)
            shrink = apply_shrink.get(plan_id, 0)
            if plan_id in placeholder_ids:
                # 占位已发权益：把占用计入组件扣减，用来解释争用，但不进报告行。
                if alloc.allocated < count:
                    warnings.append(
                        f"{store} 已发 {bundle_id} 需要 {count} 张，只能兑现 {alloc.allocated} 张；"
                        "已发权益不可撤销，需补货而非缩量。"
                    )
                continue
            if store not in reported_stores:
                # 这一门店只是被占位，没有试算行，不生成报告。
                continue
            if locked:
                target = count
                if shrink:
                    warnings.append(
                        f"{plan_id} 为已发权益，{shrink} 张缩量转给后续投放，不追溯撤销。"
                    )
            else:
                target = max(0, count - shrink)
            line_alloc = min(alloc.allocated, target)
            if locked and alloc.allocated < count:
                warnings.append(
                    f"{store} 已发 {bundle_id} 需要 {count} 张，只能兑现 {alloc.allocated} 张；"
                    "已发权益不可撤销，需补货而非缩量。"
                )
            notes = []
            if line_alloc < target:
                notes.append(f"库存或共享商品只允许 {line_alloc} 张")
            if shrink and not locked:
                notes.append(f"按新证据缩减 {shrink} 张")
            if not check.passes:
                warnings.append(
                    f"{bundle_id} 最不利替换后每张净收益 {check.worst_margin} 低于下限 {check.floor}"
                )
            lines[store].append(
                PlanLine(
                    plan_id=plan_id,
                    bundle=bundle_id,
                    store=store,
                    requested=count,
                    allocated=line_alloc,
                    status="issued" if locked else "proposed",
                    locked=locked,
                    margin_ok=check.passes,
                    notes=tuple(notes),
                )
            )
        allocations.extend(allocs)
    plan_map = component_plan(packet, allocations)
    pressure = shared_component_pressure(packet, allocations)
    overlap = overlap_from_design(packet, as_of)
    revenue_ok = all(line.margin_ok for store in lines.values() for line in store)
    reserved_ok = True
    for store, allocs in _group_by_store(allocations).items():
        state = store_state(packet, store, as_of)
        used = plan_map.get(store, {})
        for sku in state.physical:
            if used.get(sku, 0) > state.physical[sku]:
                reserved_ok = False
                warnings.append(f"{store} {sku} 扣减超过可动用的实物量")
    return TrialResult(
        store_results=lines,
        allocations=allocations,
        component_plan=plan_map,
        pressure=pressure,
        overlap=overlap,
        warnings=warnings,
        revenue_floor_ok=revenue_ok,
        reserved_intact=reserved_ok,
    )


def _group_by_store(allocs: list[Allocation]) -> dict[str, list[Allocation]]:
    result: dict[str, list[Allocation]] = {}
    for a in allocs:
        result.setdefault(a.store, []).append(a)
    return result


def _default_store(packet: Packet, bundle_id: str) -> str:
    for plan in packet.plans:
        if plan.bundle == bundle_id:
            return plan.store
    stores = {cell.store for cell in packet.stock}
    if len(stores) != 1:
        raise ValueError(f"无法确定 {bundle_id} 的门店")
    return stores.pop()


@dataclass(frozen=True)
class SwapView:
    """下一轮改一个组合时的对比视图。"""

    stored_bundle: str
    new_bundle: str
    store: str
    replaced_increment: float          # 被替代的预估增量
    replaced_range: tuple[float, float]
    new_increment: float
    new_range: tuple[float, float]
    net_change: float
    shared_components: dict[str, float]  # 仍在争用的 SKU 压力
    blocked_units: int                  # 即使收益上升也无法批的张数
    block_reasons: tuple[str, ...]
    revenue_floor_ok: bool


def swap_view(
    packet: Packet,
    plan_id: str,
    new_bundle_id: str,
    *,
    new_limit: int | None = None,
    as_of: str | None = None,
) -> SwapView:
    """把原计划换成另一组合，回答三件事。

    1. 哪段预估代替了原来的消费（旧组合增量 → 新组合增量）；
    2. 哪一处仍在抢共享商品（各 SKU 的申请量 / 可动用实物）；
    3. 即使预计收益上升，为什么仍有一部分发券量不能批（库存 / 会员保留 /
       共享商品 / 收益下限）。
    """
    from .design import estimate_rotation

    old_plan = packet.plan_by_id[plan_id]
    store = old_plan.store
    design = estimate_rotation(packet, as_of)

    def effect_range(bundle_id: str) -> tuple[float, tuple[float, float]]:
        e = design.effect(bundle_id) if bundle_id in ("C1", "C3") else design.effect("C1+C3")
        return e.per_window, (e.ci_low, e.ci_high)

    old_point, old_range = effect_range(old_plan.bundle)
    new_point, new_range = effect_range(new_bundle_id)
    limit = new_limit if new_limit is not None else old_plan.limit

    result = trial(packet, [(plan_id, new_bundle_id, limit)], as_of=as_of)
    allocated = result.store_results[store][0].allocated if store in result.store_results else 0
    reasons = []
    if allocated < limit:
        state = store_state(packet, store, as_of)
        # 找出卡在哪件商品。
        bundle = packet.bundle_by_id[new_bundle_id]
        for sku, per in bundle.components.items():
            need = per * limit
            pool = state.physical.get(sku, 0) + sum(
                state.physical.get(alt, 0) for alt in bundle.alt.get(sku, [])
            )
            if need > pool:
                reasons.append(f"{store} {sku} 需求 {need} 超过可动用 {pool}")
        if not bundle.components:
            reasons.append("组合没有可用组件")
        reasons.append("已发权益不可撤销，先占部分实物")
    check = margin_check(packet, new_bundle_id)
    if not check.passes:
        reasons.append(f"最不利替换后净收益 {check.worst_margin} 低于下限 {check.floor}")
    shared = {
        sku: result.pressure.get(store, {}).get(sku, 0.0) for sku in result.pressure.get(store, {})
    }
    return SwapView(
        stored_bundle=old_plan.bundle,
        new_bundle=new_bundle_id,
        store=store,
        replaced_increment=old_point,
        replaced_range=old_range,
        new_increment=new_point,
        new_range=new_range,
        net_change=new_point - old_point,
        shared_components=shared,
        blocked_units=max(0, limit - allocated),
        block_reasons=tuple(dict.fromkeys(reasons)),
        revenue_floor_ok=check.passes,
    )

    plan_id: str
    store: str
    original: int
    locked_keep: int
    shrinkable: int
    proposed_shrink: int
    remaining: int
    reason: str


def shrink_plan(packet: Packet, plan_id: str, shrink_by: int, reason: str) -> ShrinkPlan:
    """把缩量拆成「必须留给已领券/会员」与「可缩减的后续投放」。"""
    plan = packet.plan_by_id[plan_id]
    if plan.issued:
        # 已发权益不因重跑分析失效：全部保留，缩量只能来自后续投放。
        return ShrinkPlan(
            plan_id=plan_id,
            store=plan.store,
            original=plan.limit,
            locked_keep=plan.limit,
            shrinkable=0,
            proposed_shrink=0,
            remaining=plan.limit,
            reason=f"已发权益不追溯撤销；{reason}只能用于缩减后续投放",
        )
    shrink = max(0, min(shrink_by, plan.limit))
    return ShrinkPlan(
        plan_id=plan_id,
        store=plan.store,
        original=plan.limit,
        locked_keep=0,
        shrinkable=plan.limit,
        proposed_shrink=shrink,
        remaining=plan.limit - shrink,
        reason=reason,
    )
