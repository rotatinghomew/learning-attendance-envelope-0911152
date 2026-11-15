import copy
import json
from pathlib import Path
import unittest

from learning.codec import digest, encode
from learning.domain import ANALYSES, validate
from learning.occupancy import (
    DayModel,
    EvidenceIndex,
    OccupancyError,
    apply_scenario,
    build_segments,
    capacity_diagnostics,
    clipped_scan,
    forced_minutes,
    normalize_parameters,
    occupancy_bounds,
    structural_diagnostics,
)


def load(release="term-a"):
    return json.loads(Path("data/releases.json").read_text(encoding="utf-8"))[release]


class OccupancyAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load("term-a")
        validate(cls.tables)
        cls.result = ANALYSES["occupancy_bounds"](cls.tables, {})
        cls.baseline = cls.result["baseline"]

    def test_registered_and_serializable(self):
        self.assertIn("occupancy_bounds", ANALYSES)
        self.assertEqual(self.result["analysis"], "occupancy_bounds")
        self.assertTrue(self.result["assumptions"])
        encode(self.result)  # 不允许 NaN/Infinity，且可 JSON 序列化

    def test_does_not_mutate_evidence(self):
        before = digest(self.tables)
        ANALYSES["occupancy_bounds"](self.tables, {})
        self.assertEqual(before, digest(self.tables))

    def test_order_independence(self):
        reordered = {name: list(reversed(rows)) for name, rows in self.tables.items()}
        second = ANALYSES["occupancy_bounds"](reordered, {})
        self.assertEqual(digest(self.result), digest(second))

    def test_term_a_and_term_b_agree(self):
        result_b = ANALYSES["occupancy_bounds"](load("term-b"), {})
        self.assertEqual(encode(self.result), encode(result_b))

    def test_basic_bound_relationships(self):
        baseline = self.baseline
        self.assertTrue(baseline["feasible"])
        self.assertEqual(len(baseline["sessions"]), len(self.tables["sessions"]))
        for row in baseline["sessions"]:
            self.assertLessEqual(row["scan_observed"], row["headcount_lower"])
            self.assertLessEqual(row["headcount_lower"], row["headcount_upper"])
            self.assertLessEqual(row["headcount_upper"], row["booked"])
            self.assertLessEqual(0, row["seat_minutes_lower"])
            self.assertLessEqual(row["seat_minutes_lower"], row["seat_minutes_upper"])
            self.assertLessEqual(row["seat_minutes_upper"], row["seat_minutes_capacity"])
            self.assertIsNotNone(row["utilization_lower"])
            self.assertGreaterEqual(row["utilization_upper"], row["utilization_lower"])
        for segment in baseline["segments"]:
            self.assertLessEqual(0, segment["peak_lower"])
            self.assertLessEqual(segment["peak_lower"], segment["peak_upper"])
            self.assertLessEqual(segment["peak_upper"], segment["capacity"])

    def test_seat_minutes_match_independent_integer_model(self):
        # 用全整数模型独立计算代表性会话的座位分钟极值，核对 LP 路径结果。
        baseline = self.baseline
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        checked = {"s00", "s05", "s11", "s17"}
        for day in sorted({row["day"] for row in self.tables["sessions"]}):
            model = DayModel(day, self.tables, self.tables["sessions"])
            for session_id in sorted(set(model.session_rows) & checked):
                cost = np.zeros(model.nvar)
                for block in model.blocks:
                    if block["session"] == session_id:
                        cost[block["x0"]:block["x0"] + block["n"]] = 1.0
                row = next(item for item in baseline["sessions"]
                           if item["session"] == session_id)
                for sense, sign, key in (
                        ("min", 1.0, "seat_minutes_lower"),
                        ("max", -1.0, "seat_minutes_upper")):
                    result = milp(
                        c=sign * cost,
                        constraints=LinearConstraint(model.matrix, model.lower, model.upper),
                        integrality=np.ones(model.nvar), bounds=Bounds(0, 1),
                        options={"time_limit": 60, "disp": False})
                    self.assertEqual(result.status, 0)
                    self.assertEqual(row[key], round(sign * result.fun))

    def test_evidence_is_retained(self):
        baseline = ANALYSES["occupancy_bounds"](self.tables, {})["baseline"]
        self.assertEqual(baseline["evidence"]["scan_records"], len(self.tables["scans"]))
        self.assertEqual(baseline["evidence"]["booking_records"], len(self.tables["bookings"]))
        # 每条刷卡都保留可复核的强制在场分钟。
        self.assertEqual(len(baseline["evidence"]["forced_presence"]), len(self.tables["scans"]))
        for item in baseline["evidence"]["forced_presence"]:
            self.assertIn("learner", item)
            self.assertIn("session", item)
            self.assertIsInstance(item["forced_minutes"], list)


class LimitedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tables = load("limited")
        validate(self.tables)

    def test_no_scans_widens_to_booking_bounds(self):
        baseline = ANALYSES["occupancy_bounds"](self.tables, {})["baseline"]
        self.assertTrue(baseline["feasible"])
        for row in baseline["sessions"]:
            self.assertEqual(row["scan_observed"], 0)
            self.assertEqual(row["headcount_lower"], 0)
            self.assertEqual(row["headcount_upper"], row["booked"])
            self.assertEqual(row["seat_minutes_lower"], 0)
            self.assertGreater(row["seat_minutes_upper"], 0)
        # 没有任何观测时，分段下界应为 0 并标记为证据不足/可能闲置。
        self.assertTrue(all(segment["peak_lower"] == 0 for segment in baseline["segments"]))
        self.assertTrue(
            all(segment["pressure"] in ("evidence_gap_or_idle", "possible_over_capacity")
                for segment in baseline["segments"]))


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.tables = load("term-a")
        validate(self.tables)

    def test_scenario_only_changes_room_or_capacity(self):
        parameters = {"scenario": [{"session": "s03", "room": "r9", "capacity": 30}]}
        sessions, normalized = apply_scenario(self.tables, parameters)
        changed = next(row for row in sessions if row["session"] == "s03")
        self.assertEqual(changed["room"], "r9")
        self.assertEqual(changed["capacity"], 30)
        self.assertEqual(normalized, parameters)
        # 原始证据不被修改。
        self.assertEqual(
            next(row for row in self.tables["sessions"] if row["session"] == "s03")["room"], "r0")

    def test_feasible_scenario_comparison(self):
        result = occupancy_bounds(
            self.tables, {"scenario": [{"session": "s03", "room": "r9", "capacity": 30}]})
        comparison = result["comparison"]
        self.assertTrue(comparison["baseline_feasible"])
        self.assertTrue(comparison["scenario_feasible"])
        row = next(item for item in comparison["sessions"] if item["session"] == "s03")
        self.assertTrue(row["room_changed"])
        self.assertEqual(row["capacity_change"], 12)
        self.assertIsNotNone(row["utilization_upper_change"])
        self.assertLess(row["utilization_upper_change"], 0)

    def test_tight_capacity_is_contradictory_and_located(self):
        result = occupancy_bounds(
            self.tables, {"scenario": [{"session": "s03", "capacity": 1}]})
        scenario = result["scenario"]
        self.assertFalse(scenario["feasible"])
        self.assertTrue(scenario["contradictions"])
        first = scenario["contradictions"][0]
        self.assertEqual(first["room"], "r0")
        self.assertGreater(first["worst_forced"], first["capacity"])
        self.assertIn("overloaded_minutes", first)
        self.assertTrue(first["learner"])

    def test_invalid_parameters_rejected(self):
        bad_parameters = [
            None,
            {"scenario": {}},
            {"scenario": [{"session": "s00"}]},
            {"scenario": [{"session": "s00", "capacity": 0}]},
            {"scenario": [{"session": "s00", "capacity": True}]},
            {"scenario": [{"session": "s00", "room": ""}]},
            {"scenario": [{"session": "s00", "evidence": "changed"}]},
            {"scenario": [{"session": "s00", "capacity": 5},
                          {"session": "s00", "capacity": 6}]},
        ]
        for parameters in bad_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(OccupancyError):
                    normalize_parameters(parameters)
        # 未知会话需要对照资料表，在完整分析入口处拒绝。
        with self.assertRaises(OccupancyError):
            occupancy_bounds(self.tables,
                             {"scenario": [{"session": "missing", "capacity": 3}]})

    def test_default_parameters(self):
        self.assertEqual(normalize_parameters({}), {"scenario": []})


class ContradictionTests(unittest.TestCase):
    def setUp(self):
        self.tables = load("term-a")

    def test_scan_outside_session_window(self):
        tables = copy.deepcopy(self.tables)
        scan = tables["scans"][0]
        session = next(row for row in tables["sessions"]
                       if row["session"] == scan["session"])
        scan["enter_lo"] = scan["enter_hi"] = session["end"] + 5
        contradictions = structural_diagnostics(tables, tables["sessions"])
        self.assertTrue(any(item["learner"] == scan["learner"]
                            and item["session"] == scan["session"]
                            for item in contradictions))

    def test_overlapping_learner_scans_are_flagged(self):
        tables = copy.deepcopy(self.tables)
        # 选同一 day、时间重叠的两个会话，给同一学员制造无法错开的刷卡区间。
        sessions = sorted(tables["sessions"], key=lambda row: row["session"])
        first = next(row for row in sessions if row["day"] == 1)
        second = next(candidate for candidate in sessions
                      if candidate["day"] == 1 and candidate["session"] != first["session"]
                      and candidate["start"] < first["end"]
                      and first["start"] < candidate["end"])
        learner = "zz-conflict"
        tables["bookings"].extend([
            {"learner": learner, "session": first["session"]},
            {"learner": learner, "session": second["session"]},
        ])
        overlap_start = max(first["start"], second["start"])
        tables["scans"].extend([
            {"learner": learner, "session": first["session"],
             "enter_lo": overlap_start, "enter_hi": overlap_start,
             "leave_lo": overlap_start + 20, "leave_hi": overlap_start + 20},
            {"learner": learner, "session": second["session"],
             "enter_lo": overlap_start, "enter_hi": overlap_start,
             "leave_lo": overlap_start + 20, "leave_hi": overlap_start + 20},
        ])
        contradictions = structural_diagnostics(tables, tables["sessions"])
        self.assertTrue(any(item["learner"] == learner for item in contradictions))

    def test_capacity_diagnosis_identifies_forced_overload(self):
        tables = copy.deepcopy(self.tables)
        target = tables["sessions"][0]
        evidence = EvidenceIndex(tables, tables["sessions"])
        target["capacity"] = 1
        contradictions = capacity_diagnostics(tables["sessions"], evidence)
        self.assertTrue(any(item["room"] == target["room"]
                            and item["day"] == target["day"]
                            and item["worst_forced"] > 1
                            for item in contradictions))


class HelperTests(unittest.TestCase):
    def test_clipped_scan_and_forced_minutes(self):
        session = {"start": 100, "end": 200}
        scan = {"enter_lo": 90, "enter_hi": 110, "leave_lo": 150, "leave_hi": 210}
        self.assertEqual(clipped_scan(scan, session), (100, 110, 150, 200))
        self.assertEqual(forced_minutes(scan, session), set(range(110, 150)))

    def test_zero_length_scan_has_no_forced_minutes(self):
        session = {"start": 100, "end": 200}
        scan = {"enter_lo": 120, "enter_hi": 120, "leave_lo": 120, "leave_hi": 120}
        self.assertEqual(forced_minutes(scan, session), set())

    def test_segments_partition_room_windows(self):
        tables = load("term-a")
        segments = build_segments(tables["sessions"])
        self.assertTrue(segments)
        for segment in segments:
            self.assertLess(segment["start"], segment["end"])
            self.assertTrue(segment["sessions"])
            self.assertEqual(segment["capacity"], min(segment["minute_capacity"].values()))


if __name__ == "__main__":
    unittest.main()
