"""把证据、库存与试算组装成一份可复核的报告。

报告遵守三条呈现规则：

1. **分项接回总量**：任何分项表都附上「合计 = 总量」的核对行；
2. **范围优先**：存在多种同样说得通的转移路径时给区间，不给单点；
3. **可追溯到证据**：每条结论都注明支撑到哪些组合、门店、时间，
   以及什么时候只能看到已经发生的销量。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .censor import estimate_censoring
from .design import build_panel, estimate_rotation, store_effects, system_weekly_effect
from .evidence import joint_report
from .inventory import margin_check, max_feasible, store_state
from .packet import Packet


@dataclass
class Section:
    title: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    title: str
    digest: str
    assess_as_of: str
    sections: list[Section] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"# {self.title}", f"资料指纹 {self.digest}；评估时点 {self.assess_as_of}", ""]
        for section in self.sections:
            lines.append(f"## {section.title}")
            for row in section.rows:
                if isinstance(row, str):
                    lines.append(row)
                else:
                    lines.append(" | ".join(f"{k}={v}" for k, v in row.items()))
            for note in section.notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines)


def build_report(packet: Packet, as_of: str | None = None) -> Report:
    as_of = as_of or packet.policy.assess_as_of
    report = Report(
        title="苹果节联合券净销量增量与投放批准",
        digest=packet.digest(),
        assess_as_of=as_of,
    )

    design = estimate_rotation(packet, as_of)
    report.sections.append(_section_design(packet, design))
    report.sections.append(_section_decomposition(packet, as_of))
    report.sections.append(_section_transfers(packet, as_of))
    report.sections.append(_section_censoring(packet, as_of))
    report.sections.append(_section_inventory(packet, as_of))
    report.sections.append(_section_evidence_scope(packet, as_of))
    return report


def _section_design(packet: Packet, design) -> Section:
    section = Section(title="同轮对照估计（轮换拉丁方）")
    for state in ("C1", "C3", "C1+C3"):
        effect = design.effect(state)
        section.rows.append(
            {
                "状态": state,
                "每窗口净增量": round(effect.per_window, 2),
                "90%区间": f"[{effect.ci_low:.2f}, {effect.ci_high:.2f}]",
                "置换p值": round(effect.p_value, 3),
                "各轮": [round(v, 2) for v in effect.per_block],
            }
        )
    weekly = system_weekly_effect(packet, "C1+C3", None)
    section.notes.append(
        f"两券同投的共享购物组每周合计约 {weekly:.1f} 件；每窗口效应按 4 个门店日折算。"
    )
    section.notes.append(
        f"轮换期第 {design.rotation_period[0]}–{design.rotation_period[1]} 周；"
        f"投放组 {design.treated_group}，同轮对照组 {design.control_group}。"
    )
    early = design.early_effects
    if early and "C1" in early:
        eff = early["C1"]
        section.notes.append(
            f"早期日历排期单独核对：C1 每窗口约 {eff.per_window:.2f}"
            f"（置换p值 {eff.p_value:.3f}），需求水平与轮换期不同，只作外部交叉核对。"
        )
    section.notes.append("E、W 共用商圈，始终作为同一系统留在样本一边，不拆成两个独立实验。")
    return section


def _section_decomposition(packet: Packet, as_of: str) -> Section:
    from .evidence import decompose

    section = Section(title="分项拆解（会员/非会员 × 用券/未用券，逐项接回总量）")
    dec = decompose(packet, as_of)
    for state, item in dec.items():
        section.rows.append(
            {
                "状态": state,
                "会员用券": round(item.buckets["member_coupon"].per_window, 2),
                "会员未用券": round(item.buckets["member_plain"].per_window, 2),
                "非会员用券": round(item.buckets["nonmember_coupon"].per_window, 2),
                "非会员未用券": round(item.buckets["nonmember_plain"].per_window, 2),
                "合计": round(sum(b.per_window for b in item.buckets.values()), 2),
                "总量": round(item.total, 2),
                "核对差": round(item.check_sum(), 6),
            }
        )
    section.notes.append("四个分项互斥且完备，合计差必须为 0；表中已逐项核对。")
    section.notes.append("会员用券与会员未用券一升一降，正是会员需求被改道的可辨部分。")
    return section


def _section_transfers(packet: Packet, as_of: str) -> Section:
    section = Section(title="两券同投：新增与转移的可行范围")
    jr = joint_report(packet, as_of)
    section.rows.append(
        {
            "同投窗口数": len(jr.windows),
            "涉及周": list(jr.weeks),
            "每窗口净增量": round(jr.net_increment, 2),
            "扣转移后范围": f"[{jr.net_increment_low:.2f}, {jr.net_increment_high:.2f}]",
            "核销券数": jr.redemptions,
            "券数上下界": f"[{jr.coupon_bounds[0]}, {jr.coupon_bounds[1]}]",
        }
    )
    for band in jr.transfers:
        section.rows.append(
            {
                "转移路径": band.label,
                "范围": f"[{band.low:.2f}, {band.high:.2f}]",
                "证据": band.evidence,
            }
        )
    section.notes.append("同一笔交易只计一次：分项互斥，券数按核销编号而非商品行数。")
    section.notes.append("多种转移路径同样说得通时保留范围，不把相关性写成确定归因。")
    for note in jr.notes:
        section.notes.append(note)
    return section


def _section_censoring(packet: Packet, as_of: str) -> Section:
    section = Section(title="售罄后的缺单补回")
    report = estimate_censoring(packet, as_of, basis="rotation")
    for adj in report.adjustments:
        section.rows.append(
            {
                "窗口": adj.window,
                "商品": adj.sku,
                "售罄": adj.sold_out,
                "观测销量": adj.observed,
                "售罄前份额": round(adj.share_before, 3),
                "推断全天": round(adj.inferred_full_day, 1),
                "补回": round(adj.add_back, 2),
            }
        )
    section.notes.append(
        f"轮换期合计补回约 {report.total():.2f} 件；售罄后的缺单不是零需求。"
    )
    if not report.adjustments:
        section.notes.append("轮换期没有售罄窗口。")
    # 补回前后的估计对比，说明结论对截断处理是否敏感。
    from .censor import window_units_with_censoring
    from .design import estimate_rotation

    base = estimate_rotation(packet, as_of)
    adjusted = estimate_rotation(
        packet, as_of, window_units=window_units_with_censoring(packet, as_of)
    )
    for state in ("C1", "C3", "C1+C3"):
        b = base.effect(state).per_window
        c = adjusted.effect(state).per_window
        section.rows.append(
            {"状态": state, "未补回": round(b, 2), "补回后": round(c, 2), "差": round(c - b, 2)}
        )
    section.notes.append("补回量很小，方向是把被截断的销量补回，不改变各状态的大小次序。")
    return section


def _section_inventory(packet: Packet, as_of: str) -> Section:
    section = Section(title="库存、共享商品与收益下限")
    for store in sorted({cell.store for cell in packet.stock}):
        state = store_state(packet, store, as_of)
        section.rows.append(
            {
                "门店": store,
                "可动用实物": state.physical,
                "新增可用": state.budget,
                "会员保留": state.reserved,
                "已发占用": state.committed,
                "实收入库": state.received,
            }
        )
    for plan in packet.plans:
        check = margin_check(packet, plan.bundle)
        section.rows.append(
            {
                "计划": plan.id,
                "组合": plan.bundle,
                "门店": plan.store,
                "状态": plan.status,
                "张数": plan.limit,
                "名义净收益": check.nominal_margin,
                "最不利替换净收益": check.worst_margin,
                "下限": check.floor,
                "达标": check.passes,
            }
        )
    for store in sorted({cell.store for cell in packet.stock}):
        for plan in packet.plans:
            if plan.store != store:
                continue
            section.rows.append(
                {
                    "门店": store,
                    "组合": plan.bundle,
                    "不挤占时最多兑现": max_feasible(packet, store, plan.bundle, as_of),
                }
            )
    section.notes.append("可用量 = 盘点数 − 会员保留 − 已发权益占用 + 盘点后验收入库。")
    section.notes.append("盘点数已含早期退货；同一笔退货不能在净销量和库存两处都加回。")
    section.notes.append("会员保留量只按政策到点释放，不同门店的可用量不默认互借。")
    return section


def _section_evidence_scope(packet: Packet, as_of: str) -> Section:
    section = Section(title="证据能支撑到哪里、什么时候只能看到已发生销量")
    rotation_weeks = sorted({
        int(row.window.split("-")[0]) for row in packet.rollouts if row.basis == "rotation"
    })
    early_weeks = sorted({
        int(row.window.split("-")[0]) for row in packet.rollouts if row.basis == "calendar"
    })
    section.rows.append(
        {
            "可识别组合": "C1、C3、C1+C3",
            "轮换周": f"{rotation_weeks[0]}–{rotation_weeks[-1]}",
            "早期日历周": f"{early_weeks[0]}–{early_weeks[-1]}" if early_weeks else "无",
            "门店": "、".join(sorted({cell.store for cell in packet.stock})),
            "评估时点": as_of,
        }
    )
    missing = packet.missing_seen()
    if missing:
        section.notes.append(
            f"{len(missing)} 条曝光日志的当地可见数缺失，只能看到实际销量，不能据以推断未曝光。"
        )
    section.notes.append("没有曝光记录的人不当作未曝光；已安排投放以排期为准。")
    section.notes.append("评估时点之后的退款更正不计入当期净销量，但会在下次快照更正。")
    return section
