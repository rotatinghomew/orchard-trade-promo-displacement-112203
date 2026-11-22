"""核对演练资料的公开输入约定。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inspect_packet import read_packet


class PacketTest(unittest.TestCase):
    def test_related_records(self):
        doc = read_packet(Path("资料/演练.json"))
        windows={row['window'] for row in doc['listings']}
        self.assertTrue({row['window'] for row in doc['orders']+doc['exposures']}<=windows)
        self.assertTrue(len({row['id'] for row in doc['orders']})==len(doc['orders']))
        self.assertTrue({row['order'] for row in doc['updates']} <= {row['id'] for row in doc['orders']})
        self.assertTrue({row['bundle'] for row in doc['planned']} <= {row['id'] for row in doc['bundle']})
        self.assertTrue(all(not row['resellable'] or row['refund'] for row in doc['orders']))
        self.assertTrue(any(row['sold_out'] for row in doc['listings']) and any(not row['promo'] for row in doc['listings']))

    def test_rollout_and_stock_evidence_can_be_joined(self):
        packet = read_packet(Path("资料/演练.json"))
        offers = {row['window']: set(row['offered']) for row in packet['rollouts']}
        self.assertEqual(set(offers), {row['window'] for row in packet['listings']})
        blocks = {}
        for row in packet['rollouts']:
            if row['basis'] == 'rotation':
                blocks.setdefault(row['block'], set()).add(frozenset(row['offered']))
        states = {frozenset(x) for x in ((), ('C1',), ('C3',), ('C1', 'C3'))}
        self.assertGreaterEqual(sum(states <= offerings for offerings in blocks.values()),
                                packet['policy']['min_comparable_blocks'])
        lines = {row['id']: row for row in packet['orders']}
        for row in packet['redemptions']:
            self.assertIn(row['coupon'], offers[row['window']])
            self.assertTrue(all(lines[key]['coupon'] == row['coupon']
                                and lines[key]['household'] == row['household']
                                and lines[key]['window'] == row['window'] for key in row['lines']))
        updates = {row['order']: row for row in packet['updates']}
        self.assertEqual({row['accepted'] for row in packet['stock_receipts']}, {True, False})
        self.assertTrue(all(row['accepted'] == updates[row['order']]['resellable']
                            and row['sku'] == lines[row['order']]['actual']
                            and row['received'] > packet['policy']['stock_counted_at']
                            for row in packet['stock_receipts']))

    def test_reject_wrong_container(self):
        with TemporaryDirectory() as folder:
            packet = Path(folder) / "unexpected.json"
            packet.write_text(json.dumps({"invalid": 3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_packet(packet)
