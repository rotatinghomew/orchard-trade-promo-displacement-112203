"""窗口面板与联合券投放的识别。

设计事实（全部由资料本身给出，不预设结论）：

* 营业按 ``window``（周-门店-日序）记录，``week`` 的先后就是复盘先后；
* ``rollouts`` 给出每一轮实际公开的券种 ``offered``，空券种是确认没有投放，
  不能由个人曝光漏记反推；
* 轮换轮次里，同一 ``block`` 的四周恰好各出现一次
  ``无券 / 仅 C1 / 仅 C3 / 两券同投``，构成 4×4 拉丁方：状态与轮次、周内位置都正交；
* ``zone_groups`` 里同组的门店共用商圈（E、W 同组），组内不能互作独立对照；
  另一组（C）从不投放，只作同轮对照。

因此主估计量是「同一轮、同一周内位置」的对照差分，再对四个轮次取平均。
共同客流冲击由同轮 C 组吸收；E、W 属同一共享购物组，始终作为一个系统留在
同一边，不当作两个独立样本。推断用块内换序的精确置换分布，而不是把点估计
当成确定归因。
"""
from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from .packet import Packet, STATES, state_label

ROTATION = "rotation"
CALENDAR = "calendar"


CensorLookup = None  # 由 censor 模块注入，避免循环导入


@dataclass(frozen=True)
class WindowObs:
    window: str
    week: int
    day: int
    zone: str
    zone_group: str
    block: str
    position: int | None
    basis: str
    state: str
    net_units: int


@dataclass(frozen=True)
class Cell:
    """一个轮次×周内位置上的同轮对照差分。"""

    block: str
    position: int
    week: int
    state: str
    treated_mean: float
    control_mean: float
    treated_windows: tuple[str, ...]
    control_windows: tuple[str, ...]

    @property
    def gap(self) -> float:
        return self.treated_mean - self.control_mean


@dataclass(frozen=True)
class Effect:
    state: str
    per_window: float
    per_block: tuple[float, ...]
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    null_low: float
    null_high: float
    blocks: int


@dataclass
class DesignResult:
    cells: list[Cell]
    effects: dict[str, Effect]
    treated_group: str
    control_group: str
    rotation_period: tuple[int, int]
    early_effects: dict[str, Effect] | None = None

    def effect(self, state: str) -> Effect:
        return self.effects[state]


def _week_of(window: str) -> int:
    return int(window.split("-")[0])


def _zone_of(window: str) -> str:
    return window.split("-")[1]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _t90(df: int) -> float:
    # 双侧 90% 的 t 临界值（df<=6 时逐一列出，避免依赖第三方库）。
    table = {1: 6.3138, 2: 2.92, 3: 2.3534, 4: 2.1318, 5: 2.015, 6: 1.9432,
             7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125}
    return table.get(df, 1.645)


def _t95(df: int) -> float:
    table = {1: 12.706, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
             7: 2.3646, 8: 2.306, 9: 2.2622, 10: 2.2281}
    return table.get(df, 1.96)


def treated_and_control_groups(packet: Packet) -> tuple[str, str]:
    """从排期本身推断哪个商圈组被投放、哪个组只作对照。"""
    offered_groups = {
        packet.zone_group(zone)
        for row in packet.rollouts
        if row.offered
        for zone in row.eligible
    }
    if len(offered_groups) != 1:
        raise ValueError(f"无法唯一确定投放商圈组：{sorted(offered_groups)}")
    treated = offered_groups.pop()
    all_groups = {packet.zone_group(z) for row in packet.rollouts for z in row.eligible}
    control = all_groups - {treated}
    if len(control) != 1:
        raise ValueError("对照商圈组不唯一，无法做同轮对照")
    return treated, control.pop()


def build_panel(packet: Packet, as_of: str | None = None) -> list[WindowObs]:
    """把一个窗口整理成一行；``position`` 只在轮换轮次里有意义。"""
    rotation_blocks: dict[str, list[int]] = {}
    for row in packet.rollouts:
        if row.basis == ROTATION:
            rotation_blocks.setdefault(row.block, [])
            week = _week_of(row.window)
            if week not in rotation_blocks[row.block]:
                rotation_blocks[row.block].append(week)
    position: dict[tuple[str, int], int] = {}
    for block, weeks in rotation_blocks.items():
        for idx, week in enumerate(sorted(weeks)):
            position[(block, week)] = idx

    treat_group, _ = treated_and_control_groups(packet)
    rows: list[WindowObs] = []
    seen: set[str] = set()
    for window, listing in packet.listings_by_window.items():
        if window in seen:
            continue
        seen.add(window)
        rollout = packet.rollout_by_window[window]
        week = _week_of(window)
        zone = _zone_of(window)
        rows.append(
            WindowObs(
                window=window,
                week=week,
                day=int(window.split("-")[2]),
                zone=zone,
                zone_group=packet.zone_group(zone),
                block=rollout.block,
                position=position.get((rollout.block, week)),
                basis=rollout.basis,
                state=rollout.state,
                net_units=packet.window_net_units(window, as_of),
            )
        )
    rows.sort(key=lambda r: (r.week, r.zone, r.day))
    return rows


def _week_state(packet: Packet, week: int, block: str, treat_group: str) -> str:
    states = {
        packet.rollout_by_window[w].state
        for w in packet.rollout_by_window
        if _week_of(w) == week
        and packet.rollout_by_window[w].block == block
        and packet.zone_group(_zone_of(w)) == treat_group
    }
    if len(states) != 1:
        raise ValueError(f"第 {week} 周投放状态不一致：{states}")
    return states.pop()


def rotation_cells(
    packet: Packet,
    as_of: str | None = None,
    window_units: dict[str, int] | None = None,
) -> list[Cell]:
    """构造轮换轮次的同轮对照单元。"""
    treat_group, control_group = treated_and_control_groups(packet)
    units = window_units or {
        window: packet.window_net_units(window, as_of) for window in packet.rollout_by_window
    }
    by_block: dict[str, list[int]] = {}
    for row in packet.rollouts:
        if row.basis == ROTATION:
            by_block.setdefault(row.block, [])
            week = _week_of(row.window)
            if week not in by_block[row.block]:
                by_block[row.block].append(week)

    cells: list[Cell] = []
    for block in sorted(by_block):
        for position, week in enumerate(sorted(by_block[block])):
            treated = [
                w for w in units
                if _week_of(w) == week
                and packet.rollout_by_window[w].block == block
                and packet.zone_group(_zone_of(w)) == treat_group
            ]
            control = [
                w for w in units
                if _week_of(w) == week
                and packet.rollout_by_window[w].block == block
                and packet.zone_group(_zone_of(w)) == control_group
            ]
            cells.append(
                Cell(
                    block=block,
                    position=position,
                    week=week,
                    state=_week_state(packet, week, block, treat_group),
                    treated_mean=_mean(units[w] for w in treated),
                    control_mean=_mean(units[w] for w in control),
                    treated_windows=tuple(sorted(treated)),
                    control_windows=tuple(sorted(control)),
                )
            )
    return cells


def _observed_effects(cells: list[Cell]) -> dict[str, tuple[float, tuple[float, ...]]]:
    """每个状态的效应 = 各轮次内 (gap(state) − gap(none)) 的平均。

    拉丁方的正交性保证这等于对轮次和周内位置做双向固定效应的 OLS 估计；
    模块测试会在数据上核对这一点。
    """
    by_block: dict[str, dict[str, float]] = {}
    for cell in cells:
        by_block.setdefault(cell.block, {})[cell.state] = cell.gap
    result: dict[str, tuple[float, tuple[float, ...]]] = {}
    for state in STATES:
        per_block = []
        for block in sorted(by_block):
            if state in by_block[block] and "none" in by_block[block]:
                per_block.append(by_block[block][state] - by_block[block]["none"])
        if per_block:
            result[state] = (_mean(per_block), tuple(per_block))
    return result


def _state_positions(cells: list[Cell]) -> dict[str, dict[str, int]]:
    mapping: dict[str, dict[str, int]] = {}
    for cell in cells:
        mapping.setdefault(cell.block, {})[cell.state] = cell.position
    return mapping


def permutation_null(cells: list[Cell], alpha: float = 0.05) -> list[float]:
    """块内换序的精确零分布（与具体状态无关）。

    原假设下，状态标签在每轮的四个周内位置之间等可能换序。要计算的统计量
    是「标记为某状态的位置 − 标记为 none 的位置」的块内均值。换序后，
    这两个位置就是四个位置里一对有序的不同位置，每对等概率（1/12）。
    各轮相互独立，所以零分布 = 每轮 12 个差值的均值的卷积。

    这样既保留「同一轮促销一起留在样本一边」，又比枚举 24^4 个排列快得多。
    """
    blocks = sorted({cell.block for cell in cells})
    per_block: list[list[float]] = []
    for block in blocks:
        gaps = [0.0] * 4
        for cell in cells:
            if cell.block == block:
                gaps[cell.position] = cell.gap
        diffs = [gaps[i] - gaps[j] for i in range(4) for j in range(4) if i != j]
        per_block.append(diffs)
    if not per_block:
        return [0.0]
    # 逐轮卷积，最后除以轮数得到均值分布。
    acc: dict[float, int] = {0.0: 1}
    for diffs in per_block:
        nxt: dict[float, int] = {}
        for base, count in acc.items():
            for value in diffs:
                key = round(base + value, 9)
                nxt[key] = nxt.get(key, 0) + count
        acc = nxt
    n_blocks = len(per_block)
    draws: list[float] = []
    for total, count in acc.items():
        draws.extend([total / n_blocks] * count)
    draws.sort()
    return draws


def permutation_p_value(draws: list[float], observed: float) -> float:
    n = len(draws)
    if not n:
        return 1.0
    return sum(1 for value in draws if value >= observed) / n


def null_interval(draws: list[float], alpha: float = 0.05) -> tuple[float, float]:
    n = len(draws)
    if not n:
        return 0.0, 0.0
    low = draws[min(n - 1, int(math.floor(alpha * n)))]
    high = draws[min(n - 1, int(math.ceil((1 - alpha) * n)) - 1)]
    return low, high


def _effect_with_inference(
    cells: list[Cell], state: str, *, two_sided: bool = False
) -> Effect:
    observed_map = _observed_effects(cells)
    point, per_block = observed_map[state]
    n = len(per_block)
    se = statistics.stdev(per_block) / math.sqrt(n) if n > 1 else float("inf")
    crit = _t95(n - 1) if two_sided else _t90(n - 1)
    draws = permutation_null(cells)
    p_value = permutation_p_value(draws, point)
    null_low, null_high = null_interval(draws)
    return Effect(
        state=state,
        per_window=point,
        per_block=per_block,
        se=se,
        ci_low=point - crit * se,
        ci_high=point + crit * se,
        p_value=p_value,
        null_low=null_low,
        null_high=null_high,
        blocks=n,
    )


def estimate_rotation(
    packet: Packet,
    as_of: str | None = None,
    window_units: dict[str, int] | None = None,
) -> DesignResult:
    cells = rotation_cells(packet, as_of, window_units)
    effects = {
        state: _effect_with_inference(cells, state)
        for state in ("C1", "C3", "C1+C3")
    }
    treat_group, control_group = treated_and_control_groups(packet)
    weeks = [cell.week for cell in cells]
    result = DesignResult(
        cells=cells,
        effects=effects,
        treated_group=treat_group,
        control_group=control_group,
        rotation_period=(min(weeks), max(weeks)),
    )
    result.early_effects = estimate_early(packet, as_of, window_units)
    return result


def estimate_early(
    packet: Packet,
    as_of: str | None = None,
    window_units: dict[str, int] | None = None,
) -> dict[str, Effect] | None:
    """早期日历排期的 C1 对照估计。

    早期各轮只上一款券，且与轮换期需求水平、售罄比例都不同，只作外部
    交叉核对，不与轮换期直接合并。
    """
    treat_group, control_group = treated_and_control_groups(packet)
    units = window_units or {
        window: packet.window_net_units(window, as_of) for window in packet.rollout_by_window
    }
    weeks = sorted({_week_of(r.window) for r in packet.rollouts if r.basis == CALENDAR})
    if not weeks:
        return None
    states = {}
    gaps = {}
    for week in weeks:
        all_windows = [
            w for w in units
            if _week_of(w) == week and packet.rollout_by_window[w].basis == CALENDAR
        ]
        if not all_windows:
            continue
        treated = [w for w in all_windows if packet.zone_group(_zone_of(w)) == treat_group]
        control = [w for w in all_windows if packet.zone_group(_zone_of(w)) == control_group]
        if not treated or not control:
            continue
        # 状态由被投放组的排期决定；对照组的 offered 恒为空，不能拿来判状态。
        states[week] = _week_state(packet, week, packet.rollout_by_window[treated[0]].block, treat_group)
        gaps[week] = _mean(units[w] for w in treated) - _mean(units[w] for w in control)

    treated_weeks = [w for w in gaps if states[w] == "C1"]
    control_weeks = [w for w in gaps if states[w] == "none"]
    if not treated_weeks or not control_weeks:
        return None
    point = _mean(gaps[w] for w in treated_weeks) - _mean(gaps[w] for w in control_weeks)

    # 对「哪几周被投放」做精确置换，样本保持为整周、整轮。
    all_weeks = treated_weeks + control_weeks
    draws = []
    for combo in itertools.combinations(all_weeks, len(treated_weeks)):
        rest = [w for w in all_weeks if w not in combo]
        draws.append(_mean(gaps[w] for w in combo) - _mean(gaps[w] for w in rest))
    draws.sort()
    n = len(draws)
    p_value = sum(1 for v in draws if v >= point) / n
    low = draws[int(math.floor(0.05 * n))]
    high = draws[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    # 轮次层面的离散度（4 个投放轮 + 4 个对照轮）
    per_block = tuple(
        gaps[w] for w in treated_weeks
    )
    se = statistics.stdev(per_block) / math.sqrt(len(per_block)) if len(per_block) > 1 else float("inf")
    return {
        "C1": Effect(
            state="C1",
            per_window=point,
            per_block=per_block,
            se=se,
            ci_low=point - _t90(len(per_block) - 1) * se,
            ci_high=point + _t90(len(per_block) - 1) * se,
            p_value=p_value,
            null_low=low,
            null_high=high,
            blocks=len(treated_weeks) + len(control_weeks),
        )
    }


def store_effects(
    packet: Packet,
    store: str,
    as_of: str | None = None,
    window_units: dict[str, int] | None = None,
) -> dict[str, Effect]:
    """只用某一门店的窗口重算轮换效应，供单店投放参考。

    E、W 同属一个共享购物组，单店数字只作描述，不当作两个独立实验。
    """
    treat_group, control_group = treated_and_control_groups(packet)
    units = window_units or {
        window: packet.window_net_units(window, as_of) for window in packet.rollout_by_window
    }
    by_block: dict[str, list[int]] = {}
    for row in packet.rollouts:
        if row.basis == ROTATION:
            by_block.setdefault(row.block, [])
            week = _week_of(row.window)
            if week not in by_block[row.block]:
                by_block[row.block].append(week)
    cells: list[Cell] = []
    for block in sorted(by_block):
        for position, week in enumerate(sorted(by_block[block])):
            if _zone_of(next(w for w in units if _week_of(w) == week)) != store:
                # 该周是否包含此门店
                if not any(_week_of(w) == week and _zone_of(w) == store for w in units):
                    continue
            treated = [w for w in units if _week_of(w) == week and _zone_of(w) == store]
            control = [
                w for w in units
                if _week_of(w) == week
                and packet.zone_group(_zone_of(w)) == control_group
            ]
            cells.append(
                Cell(
                    block=block,
                    position=position,
                    week=week,
                    state=_week_state(packet, week, block, treat_group),
                    treated_mean=_mean(units[w] for w in treated),
                    control_mean=_mean(units[w] for w in control),
                    treated_windows=tuple(sorted(treated)),
                    control_windows=tuple(sorted(control)),
                )
            )
    return {
        state: _effect_with_inference(cells, state)
        for state in ("C1", "C3", "C1+C3")
    }


def system_weekly_effect(
    packet: Packet,
    state: str,
    as_of: str | None = None,
    window_units: dict[str, int] | None = None,
) -> float:
    """共享购物组整体的每周增量（E+W 两个门店、两个日序合计）。"""
    treat_group, control_group = treated_and_control_groups(packet)
    units = window_units or {
        window: packet.window_net_units(window, as_of) for window in packet.rollout_by_window
    }
    by_block: dict[str, list[int]] = {}
    for row in packet.rollouts:
        if row.basis == ROTATION:
            by_block.setdefault(row.block, [])
            week = _week_of(row.window)
            if week not in by_block[row.block]:
                by_block[row.block].append(week)
    per_block = []
    for block in sorted(by_block):
        gaps = {}
        for week in by_block[block]:
            treated = _mean(
                units[w] for w in units
                if _week_of(w) == week and packet.zone_group(_zone_of(w)) == treat_group
            )
            control = _mean(
                units[w] for w in units
                if _week_of(w) == week and packet.zone_group(_zone_of(w)) == control_group
            )
            gaps[week] = (treated - control) * 4  # 每周 4 个门店日
        state_none = [
            gaps[w] for w in gaps
            if _week_state(packet, w, block, treat_group) == "none"
        ]
        state_x = [
            gaps[w] for w in gaps
            if _week_state(packet, w, block, treat_group) == state
        ]
        if state_none and state_x:
            per_block.append(state_x[0] - state_none[0])
    return _mean(per_block)
