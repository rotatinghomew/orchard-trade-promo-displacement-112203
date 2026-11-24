"""经办人命令行入口。

常用命令::

    python3 -m apple_promo.cli report
    python3 -m apple_promo.cli trial --plan PL3
    python3 -m apple_promo.cli approve --label v1 --plan PL3 --net 5.9
    python3 -m apple_promo.cli restore --label v1
    python3 -m apple_promo.cli diff --before v1 --after v2
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .ledger import Ledger, Snapshot, SnapshotLine, now_iso
from .packet import Packet, PacketError
from .report import build_report
from .scenario import overlap_from_design, shrink_plan, swap_view, trial
from .evidence import joint_report
from .design import estimate_rotation, store_effects

DEFAULT_PACKET = Path("资料/演练.json")
DEFAULT_LEDGER = Path("分析/批准快照.json")


def _load(path: Path) -> Packet:
    return Packet.load(path)


def cmd_report(args) -> int:
    packet = _load(args.packet)
    print(build_report(packet, args.as_of).to_text())
    return 0


def cmd_trial(args) -> int:
    packet = _load(args.packet)
    proposals = []
    for plan_id in args.plan:
        plan = packet.plan_by_id[plan_id]
        limit = args.limit if args.limit is not None else plan.limit
        proposals.append((plan_id, plan.bundle, limit))
    result = trial(packet, proposals, as_of=args.as_of)
    print("== 试算 ==")
    for store, lines in result.store_results.items():
        for line in lines:
            print(
                f"{store} {line.plan_id} {line.bundle} 申请 {line.requested} "
                f"兑现 {line.allocated} 状态 {line.status}"
                + (f" 说明 {'；'.join(line.notes)}" if line.notes else "")
            )
    print("组件扣减", result.component_plan)
    print("共享商品压力", {k: {s: round(v, 2) for s, v in m.items()} for k, m in result.pressure.items()})
    if result.overlap:
        ov = result.overlap
        print(
            f"两组同投：同投 {ov.joint_effect:.2f}，各自单投之和 {ov.additive_sum:.2f}，"
            f"重复计入范围 [{ov.low:.2f}, {ov.high:.2f}]"
        )
    for warning in result.warnings:
        print("注意:", warning)
    print("收益下限守住:", result.revenue_floor_ok, "会员保留量未侵占:", result.reserved_intact)
    return 0


def cmd_shrink(args) -> int:
    packet = _load(args.packet)
    plan = shrink_plan(packet, args.plan, args.by, args.reason)
    print(json.dumps(asdict(plan), ensure_ascii=False, indent=1))
    return 0


def cmd_approve(args) -> int:
    packet = _load(args.packet)
    jr = joint_report(packet, args.as_of)
    design = estimate_rotation(packet, args.as_of)
    lines = []
    for plan_id in args.plan:
        plan = packet.plan_by_id[plan_id]
        effect = design.effect(plan.bundle) if plan.bundle in ("C1", "C3") else design.effect("C1+C3")
        lines.append(
            SnapshotLine(
                plan_id=plan.id,
                bundle=plan.bundle,
                store=plan.store,
                limit=args.limit if args.limit is not None else plan.limit,
                status=plan.status,
                locked=plan.issued,
                allocated=args.limit if args.limit is not None else plan.limit,
                net_increment=effect.per_window,
                increment_low=jr.net_increment_low,
                increment_high=jr.net_increment_high,
            )
        )
    ledger = Ledger(args.ledger)
    snapshot = Snapshot(
        label=args.label,
        created_at=now_iso(),
        packet_digest=packet.digest(),
        assess_as_of=args.as_of or packet.policy.assess_as_of,
        lines=tuple(lines),
        context={"shared_components": {"note": "E、W 共用商圈与可替换商品"}},
    )
    ledger.add(snapshot)
    print(f"已记录批准时点 {args.label}（资料指纹 {snapshot.packet_digest}）")
    return 0


def cmd_restore(args) -> int:
    packet = _load(args.packet)
    ledger = Ledger(args.ledger)
    restored = ledger.restore(args.label, packet.digest())
    snap = restored["snapshot"]
    print(f"批准时点 {snap.label}（{snap.created_at}）{restored['note']}")
    for line in snap.lines:
        print(
            f"  {line.plan_id} {line.store} {line.bundle} 张数 {line.limit} "
            f"状态 {line.status} 每窗口增量 {line.net_increment:.2f} "
            f"范围 [{line.increment_low:.2f}, {line.increment_high:.2f}]"
        )
    return 0


def cmd_diff(args) -> int:
    ledger = Ledger(args.ledger)
    result = ledger.diff(args.before, args.after)
    print(f"比较 {result['before']} → {result['after']}")
    for change in result["changed"]:
        if change["change"] == "调整":
            print(
                f"  {change['plan_id']} 调整：张数 {change['delta_limit']:+d}，"
                f"增量 {change['delta_increment']:+.2f}"
            )
        else:
            print(f"  {change['plan_id']} {change['change']}")
    if result["shared_component_notes"]:
        print("共享商品:", result["shared_component_notes"])
    return 0


def cmd_swap(args) -> int:
    packet = _load(args.packet)
    view = swap_view(packet, args.plan, args.to, new_limit=args.limit, as_of=args.as_of)
    print(f"换组合：{view.stored_bundle} → {view.new_bundle}（{view.store}）")
    print(
        f"  原预估 {view.replaced_increment:.2f} 每窗口 "
        f"区间 [{view.replaced_range[0]:.2f}, {view.replaced_range[1]:.2f}]"
    )
    print(
        f"  新预估 {view.new_increment:.2f} 每窗口 "
        f"区间 [{view.new_range[0]:.2f}, {view.new_range[1]:.2f}]  变化 {view.net_change:+.2f}"
    )
    print("  仍在争用的共享商品:", {k: round(v, 2) for k, v in view.shared_components.items()})
    if view.blocked_units:
        print(f"  即使收益上升仍有 {view.blocked_units} 张不能批：")
        for reason in view.block_reasons:
            print("   -", reason)
    print("  收益下限守住:", view.revenue_floor_ok)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="苹果节联合券分析")
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--as-of", default=None, help="评估时点，默认取资料内 policy.assess_as_of")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("report", help="输出完整报告").set_defaults(func=cmd_report)

    p_trial = sub.add_parser("trial", help="试算给定计划")
    p_trial.add_argument("--plan", nargs="+", required=True)
    p_trial.add_argument("--limit", type=int, default=None)
    p_trial.set_defaults(func=cmd_trial)

    p_shrink = sub.add_parser("shrink", help="按新证据拆解缩量")
    p_shrink.add_argument("--plan", required=True)
    p_shrink.add_argument("--by", type=int, required=True)
    p_shrink.add_argument("--reason", default="新证据")
    p_shrink.set_defaults(func=cmd_shrink)

    p_approve = sub.add_parser("approve", help="记录批准时点快照")
    p_approve.add_argument("--label", required=True)
    p_approve.add_argument("--plan", nargs="+", required=True)
    p_approve.add_argument("--limit", type=int, default=None)
    p_approve.set_defaults(func=cmd_approve)

    p_swap = sub.add_parser("swap", help="改一个组合后的对比视图")
    p_swap.add_argument("--plan", required=True)
    p_swap.add_argument("--to", required=True)
    p_swap.add_argument("--limit", type=int, default=None)
    p_swap.set_defaults(func=cmd_swap)

    p_restore = sub.add_parser("restore", help="恢复批准时点")
    p_restore.add_argument("--label", required=True)
    p_restore.set_defaults(func=cmd_restore)

    p_diff = sub.add_parser("diff", help="比较两个批准时点")
    p_diff.add_argument("--before", required=True)
    p_diff.add_argument("--after", required=True)
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (PacketError, KeyError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
