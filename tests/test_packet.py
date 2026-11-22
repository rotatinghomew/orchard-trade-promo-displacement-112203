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

    def test_reject_wrong_container(self):
        with TemporaryDirectory() as folder:
            packet = Path(folder) / "unexpected.json"
            packet.write_text(json.dumps({"invalid": 3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_packet(packet)
