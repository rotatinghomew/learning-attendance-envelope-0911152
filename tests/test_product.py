from collections import Counter
import json
from pathlib import Path
import unittest

from learning.domain import overview, validate


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]

    def test_course_population_reconciles(self):
        result = overview(self.tables, {})
        self.assertEqual(sum(row["booked"] for row in result["courses"]), len(self.tables["bookings"]))
        self.assertEqual(sum(row["scanned"] for row in result["courses"]), len(self.tables["scans"]))
        self.assertEqual(result["learners"], len({row["learner"] for row in self.tables["bookings"]}))

    def test_no_scan_is_not_a_deleted_booking(self):
        self.tables["scans"] = []
        validate(self.tables)
        result = overview(self.tables, {})
        self.assertTrue(all(row["scanned"] == 0 for row in result["courses"]))
        self.assertGreater(sum(row["booked"] for row in result["courses"]), 0)

    def test_inverted_scan_interval_rejected(self):
        self.tables["scans"][0]["enter_lo"] = self.tables["scans"][0]["enter_hi"] + 1
        with self.assertRaises(ValueError):
            validate(self.tables)
