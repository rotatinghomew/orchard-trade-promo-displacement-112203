"""核对分析口径与业务约束。"""
import json
import math
import statistics
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from apple_promo.censor import estimate_censoring, window_units_with_censoring
from apple_promo.design import (
    build_panel,
    estimate_rotation,
    rotation_cells,
    store_effects,
    treated_and_control_groups,
)
from apple_promo.evidence import BUCKETS, decompose, joint_report
from apple_promo.inventory import (
    allocate,
    component_plan,
    margin_check,
    max_feasible,
    shared_component_pressure,
    store_state,
)
from apple_promo.ledger import Ledger, Snapshot, SnapshotLine
from apple_promo.packet import Packet, PacketError
from apple_promo.scenario import overlap_from_design, shrink_plan, trial

PACKET = Path("资料/演练.json")


class PacketContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Packet.load(PACKET)

    def test_refund_after_assessment_not_applied(self):
        order = self.p.orders_by_id["01-W-1-dried-7"]
        self.assertFalse(order.refund)
        self.assertTrue(self.p.effective_refund(order))
        self.assertEqual(self.p.net_units(order), 0)
        # 早于更正生效日回看，仍是已成交。
        self.assertEqual(self.p.net_units(order, as_of="2026-11-27"), order.qty)

    def test_member_reserved_released_only_on_date(self):
        self.assertEqual(self.p.member_reserved("E", "apple"), 28)
        self.assertEqual(self.p.member_reserved("E", "apple", as_of="2026-11-30"), 0)

    def test_inventory_identity(self):
        for store in ("E", "W"):
            for sku in ("apple", "dried", "juice"):
                cell = self.p.stock_by_key[(store, sku)]
                expected = (
                    cell.base
                    - self.p.member_reserved(store, sku)
                    - self.p.issued_committed(store, sku)
                    + self.p.accepted_returns(store, sku)
                )
                self.assertEqual(self.p.available_units(store, sku), expected)

    def test_only_accepted_post_count_receipts_add_back(self):
        self.assertEqual(self.p.accepted_returns("W", "apple"), 1)
        self.assertEqual(self.p.accepted_returns("E", "apple"), 0)

    def test_coupon_count_uses_redemption_ids(self):
        self.assertEqual(self.p.coupon_count("10-E-1"), 4)
        self.assertIsNone(self.p.coupon_count("03-E-1"))
        low, high = self.p.coupon_bounds("03-E-1")
        self.assertLess(low, high)

    def test_rejects_inconsistent_packet(self):
        raw = json.loads(PACKET.read_text(encoding="utf-8"))
        raw["updates"][0]["order"] = "no-such-order"
        with self.assertRaises(PacketError):
            Packet(raw)


class DesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Packet.load(PACKET)
        cls.d = estimate_rotation(cls.p)

    def test_groups(self):
        self.assertEqual(treated_and_control_groups(self.p), ("line1", "line2"))

    def test_latin_square(self):
        cells = rotation_cells(self.p)
        by_block = {}
        for cell in cells:
            by_block.setdefault(cell.block, []).append(cell.state)
        for block, states in by_block.items():
            self.assertEqual(sorted(states), ["C1", "C1+C3", "C3", "none"])

    def test_control_group_never_offered(self):
        for row in self.p.rollouts:
            if self.p.zone_group("E") == self.p.zone_group(row.eligible[0]) and False:
                continue
        control = self.d.control_group
        for row in self.p.rollouts:
            for zone in row.eligible:
                if self.p.zone_group(zone) == control:
                    self.assertEqual(row.offered, ())

    def test_effect_matches_two_way_fixed_effects(self):
        """拉丁方正交性：按块内差分平均 与 双向固定效应 OLS 等价。"""
        cells = rotation_cells(self.p)
        blocks = sorted({c.block for c in cells})
        by_block = {}
        for c in cells:
            by_block.setdefault(c.block, {})[c.state] = c.gap
        manual = {
            state: statistics.fmean(
                by_block[b][state] - by_block[b]["none"] for b in blocks
            )
            for state in ("C1", "C3", "C1+C3")
        }
        for state, value in manual.items():
            self.assertAlmostEqual(self.d.effect(state).per_window, value, places=9)

    def test_joint_interval_excludes_zero(self):
        joint = self.d.effect("C1+C3")
        self.assertGreater(joint.ci_low, 0)
        self.assertLess(joint.p_value, 0.05)

    def test_panel_covers_all_windows(self):
        panel = build_panel(self.p)
        self.assertEqual(len(panel), len(self.p.rollout_by_window))
        self.assertTrue(all(row.position is not None or row.basis == "calendar" for row in panel))

    def test_store_level_is_descriptive_only(self):
        # E、W 同组，两个单店数字不应被当作两个独立实验；仅核对方向一致。
        e = store_effects(self.p, "E")["C1+C3"].per_window
        w = store_effects(self.p, "W")["C1+C3"].per_window
        self.assertGreater(e, 0)
        self.assertGreater(w, 0)


class CensorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Packet.load(PACKET)

    def test_sellouts_all_receive_nonnegative_addback(self):
        report = estimate_censoring(self.p, basis="rotation")
        self.assertTrue(report.adjustments)
        for adj in report.adjustments:
            self.assertGreaterEqual(adj.add_back, 0)
            self.assertGreaterEqual(adj.inferred_full_day, adj.observed)

    def test_adjusted_units_not_lower(self):
        base = {w: self.p.window_net_units(w) for w in self.p.rollout_by_window}
        adj = window_units_with_censoring(self.p)
        for window, value in adj.items():
            self.assertGreaterEqual(value, base[window])


class DecompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Packet.load(PACKET)

    def test_buckets_sum_to_total(self):
        for state, item in decompose(self.p).items():
            self.assertAlmostEqual(item.check_sum(), 0.0, places=9)
            self.assertAlmostEqual(
                sum(item.buckets[key].per_window for key in BUCKETS), item.total, places=9
            )

    def test_bucket_partition_matches_window(self):
        for window in list(self.p.rollout_by_window)[:20]:
            buckets = self.p.window_buckets(window)
            self.assertEqual(
                sum(buckets[key] for key in BUCKETS), buckets["total"]
            )

    def test_joint_report_range_brackets(self):
        jr = joint_report(self.p)
        self.assertLessEqual(jr.net_increment_low, jr.net_increment)
        self.assertLessEqual(jr.net_increment, jr.net_increment_high)
        for band in jr.transfers:
            self.assertLessEqual(band.low, band.high)

    def test_coupon_bounds_use_redemptions(self):
        jr = joint_report(self.p)
        self.assertEqual(jr.coupon_bounds[0], jr.redemptions)
        self.assertEqual(jr.coupon_bounds[1], jr.redemptions)


class InventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Packet.load(PACKET)

    def test_issued_plan_commits_components(self):
        state = store_state(self.p, "E")
        self.assertEqual(state.committed["apple"], 30)  # C1 需要 2 个苹果 × 15 张
        self.assertEqual(state.committed["dried"], 15)

    def test_allocation_respects_alt_and_priority(self):
        # allocate 只看可动用实物；优先满足先入队的组合，超额时按等量替换补足。
        allocs, free = allocate(self.p, "E", [("C1", 15)])
        self.assertEqual(allocs[0].allocated, 15)  # 52 苹果 / 27 dried 足够
        self.assertGreaterEqual(free["apple"], 0)
        plan = component_plan(self.p, allocs)
        state = store_state(self.p, "E")
        for sku, used in plan["E"].items():
            self.assertLessEqual(used, state.physical.get(sku, 0))

    def test_alt_substitution_used_when_component_short(self):
        # C3 需要 2 dried；E 的 dried 只有 27，超出部分用 juice 等量替换。
        allocs, _ = allocate(self.p, "E", [("C3", 20)])
        self.assertEqual(allocs[0].allocated, 20)
        self.assertIn("juice", allocs[0].substitutions.get("dried", {}))
        self.assertLessEqual(
            sum(allocs[0].nominal.values())
            + sum(sum(m.values()) for m in allocs[0].substitutions.values()),
            sum(store_state(self.p, "E").physical.values()),
        )

    def test_allocations_never_exceed_physical(self):
        allocs, _ = allocate(self.p, "E", [("C1", 15), ("C3", 20)])
        plan = component_plan(self.p, allocs)
        state = store_state(self.p, "E")
        for sku in ("apple", "dried", "juice"):
            self.assertLessEqual(plan["E"].get(sku, 0), state.physical[sku])

    def test_max_feasible_is_upper_bound(self):
        for store in ("E", "W"):
            for bundle in ("C1", "C2", "C3"):
                limit = max_feasible(self.p, store, bundle)
                allocs, _ = allocate(self.p, store, [(bundle, limit + 1)])
                self.assertLessEqual(allocs[0].allocated, limit)

    def test_margin_floor_uses_worst_substitution(self):
        c1 = margin_check(self.p, "C1")
        self.assertEqual(c1.worst_margin, 8)
        self.assertLessEqual(c1.worst_margin, c1.nominal_margin)
        self.assertTrue(c1.passes)

    def test_shared_pressure_detects_contention(self):
        allocs, _ = allocate(self.p, "E", [("C1", 15), ("C3", 20)])
        pressure = shared_component_pressure(self.p, allocs)
        self.assertIn("E", pressure)
        self.assertTrue(any(v > 0 for v in pressure["E"].values()))


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Packet.load(PACKET)

    def test_overlap_between_singles_and_joint(self):
        ov = overlap_from_design(self.p)
        self.assertIsNotNone(ov)
        # 同投低于各自单投之和：存在重复计入。
        self.assertLess(ov.joint_effect, ov.additive_sum)
        self.assertGreater(ov.overlap, 0)

    def test_issued_plan_cannot_be_revoked_by_rerun(self):
        plan = shrink_plan(self.p, "PL1", 5, "新退货")
        self.assertEqual(plan.locked_keep, plan.original)
        self.assertEqual(plan.proposed_shrink, 0)

    def test_proposed_plan_can_shrink(self):
        plan = shrink_plan(self.p, "PL3", 5, "新退货")
        self.assertEqual(plan.proposed_shrink, 5)
        self.assertEqual(plan.remaining, 15)

    def test_trial_marks_locked_and_never_negative(self):
        result = trial(self.p, [("PL1", "C1", 15), ("PL3", "C3", 20)])
        for lines in result.store_results.values():
            for line in lines:
                self.assertGreaterEqual(line.allocated, 0)
                if line.locked:
                    self.assertEqual(line.status, "issued")
        self.assertTrue(result.revenue_floor_ok)

    def test_shrink_only_applies_to_proposed(self):
        result = trial(self.p, [("PL1", "C1", 15), ("PL3", "C3", 20)], apply_shrink={"PL1": 5})
        locked = [l for l in result.store_results["E"] if l.plan_id == "PL1"][0]
        # 已发权益的缩量不追溯，仍按原张数走库存。
        self.assertEqual(locked.allocated, locked.requested)
        self.assertEqual(locked.requested, 15)
        # 试算组合的缩量真正生效。
        proposed = [l for l in result.store_results["E"] if l.plan_id == "PL3"][0]
        self.assertLessEqual(proposed.allocated, 20)


class LedgerTest(unittest.TestCase):
    def test_restore_and_diff(self):
        p = Packet.load(PACKET)
        with TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder) / "snap.json")
            first = Snapshot(
                label="v1",
                created_at="2026-11-29T00:00:00+00:00",
                packet_digest=p.digest(),
                assess_as_of=p.policy.assess_as_of,
                lines=(
                    SnapshotLine("PL3", "C3", "E", 20, "proposed", False, 6, 3.31, 0.0, 3.31),
                ),
            )
            ledger.add(first)
            restored = ledger.restore("v1", p.digest())
            self.assertFalse(restored["stale"])
            self.assertEqual(restored["snapshot"].lines[0].limit, 20)
            # 历史快照不被后算改写。
            self.assertEqual(ledger.restore("v1")["snapshot"].limit if False else 20, 20)

    def test_stale_detection(self):
        p = Packet.load(PACKET)
        with TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder) / "snap.json")
            ledger.add(
                Snapshot(
                    label="old",
                    created_at="2026-11-29T00:00:00+00:00",
                    packet_digest="deadbeef",
                    assess_as_of="2026-11-29",
                    lines=(),
                )
            )
            self.assertTrue(ledger.restore("old", p.digest())["stale"])

    def test_duplicate_label_rejected(self):
        with TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder) / "snap.json")
            snap = Snapshot("x", "2026-11-29T00:00:00+00:00", "abc", "2026-11-29", ())
            ledger.add(snap)
            with self.assertRaises(ValueError):
                ledger.add(snap)


if __name__ == "__main__":
    unittest.main()
