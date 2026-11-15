"""占用边界分析（occupancy_bounds）。

本模块在业务资料（sessions/bookings/scans）之上回答“某时段教室里到底
可能有多少人、可能产生多少座位分钟”的问题。刷卡设备只留下进、出时间的
闭区间，没有刷卡并不代表缺席，因此这里不构造唯一的到课表，而是计算每个
指标在全部可行安排下的上下界。

建模约定（与 docs/data-contract.md 一致）：
- 时间为当日零点后的分钟，课程区间左闭右开 [start, end)，day 隔离日期。
- 学员对已报名课程可以缺席；未被观察到的到课必须是课程窗口内的连续区间。
- 刷卡记录表示该学员必然到课，且入场 e、离场 l 各取自登记的闭区间，e <= l；
  其在场分钟为 [e, l)，允许 e == l 的零时长到课（计入人头、不产生座位分钟）。
- 同一学员同一分钟至多在一个教室内；同一教室同一分钟的占用不超过该室容量。
- 课程以外的在场不允许，因此刷卡边界会先裁剪到课程窗口；裁剪后无可选时间
  即构成与课程时间矛盾的记录。

两类指标的求法：
- 人头与同时占用峰值可以纯组合精确判定：学员在分钟 t 能出现在教室，当且仅当
  他报名了该分钟在该教室进行的课程、其刷卡可行窗口（若有）覆盖 t、且没有被
  另一间教室的刷卡“强制在场分钟”钉住。逐分钟枚举即可得到峰值上界，无需求解器。
- 座位分钟是跨分钟的总量，保留分钟级流守恒线性规划；松弛解出现分数时自动
  回退整数规划，并在结果中注明。

各时段/课程的极值来自各自的优化问题，通常对应不同的可行安排，结果中保留
求解依据（强制在场分钟、边界来源、矛盾诊断），不把极值拼成一张完整到课表。
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

SOLVER_TIME_LIMIT = 30.0
INTEGRAL_TOLERANCE = 1e-7

# 分析结果中用于说明口径的固定文本，便于复核。
ASSUMPTIONS = [
    "课程窗口为左闭右开的分钟区间，不同 day 之间互不占用。",
    "报名只代表到课资格，学员可缺席；未刷卡的到课必须是课程窗口内的连续区间。",
    "刷卡记录代表必然到课，入场与离场时间分别取自登记闭区间且入场不晚于离场。",
    "刷卡边界先裁剪到课程窗口；裁剪后无可行时间的记录直接判为相互矛盾。",
    "同一学员同一分钟至多占用一个教室；同一教室同一分钟占用不超过教室容量。",
    "零时长到课计入人头，不产生座位分钟。",
    "每项极值分别求解，彼此可能对应不同的可行安排，不能拼接成完整到课表。",
]


class OccupancyError(ValueError):
    """参数或场地情景不合法。"""


# --------------------------------------------------------------------------- #
# 参数与场地情景
# --------------------------------------------------------------------------- #
def normalize_parameters(parameters):
    """校验并归一化 occupancy_bounds 的参数。

    parameters 可缺省；scenario 可包含若干 sessions 覆盖项，每项只允许调整
    room 或 capacity，证据（bookings/scans）不允许改动。
    """
    if parameters is None:
        raise OccupancyError("分析参数不能为 null")
    if not isinstance(parameters, dict):
        raise OccupancyError("分析参数不是对象")
    scenario = parameters.get("scenario")
    if scenario is None:
        scenario = []
    if not isinstance(scenario, list):
        raise OccupancyError("scenario 必须是会话覆盖项数组")
    normalized = []
    seen = set()
    for item in scenario:
        if not isinstance(item, dict) or "session" not in item:
            raise OccupancyError("场地情景项缺少 session")
        session_id = item["session"]
        if not isinstance(session_id, str) or not session_id:
            raise OccupancyError("场地情景项的 session 不合法")
        if session_id in seen:
            raise OccupancyError(f"场地情景重复覆盖会话：{session_id}")
        seen.add(session_id)
        changes = {}
        if "room" in item:
            if not isinstance(item["room"], str) or not item["room"]:
                raise OccupancyError("场地情景的 room 不合法")
            changes["room"] = item["room"]
        if "capacity" in item:
            capacity = item["capacity"]
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise OccupancyError("场地情景的 capacity 必须是正整数")
            changes["capacity"] = capacity
        if not changes:
            raise OccupancyError("场地情景项未调整 room 或 capacity")
        extra = set(item) - {"session", "room", "capacity"}
        if extra:
            raise OccupancyError(f"场地情景只允许调整 room/capacity：{sorted(extra)}")
        normalized.append({"session": session_id, **changes})
    return {"scenario": normalized}


def apply_scenario(tables, parameters):
    """返回应用场地情景后的 sessions 副本；证据表保持不变。"""
    normalized = normalize_parameters(parameters)
    known = {row["session"] for row in tables["sessions"]}
    for item in normalized["scenario"]:
        if item["session"] not in known:
            raise OccupancyError(f"场地情景引用了不存在的会话：{item['session']}")
    sessions = [dict(row) for row in tables["sessions"]]
    for row in sessions:
        for item in normalized["scenario"]:
            if row["session"] == item["session"]:
                row.update({key: value for key, value in item.items() if key != "session"})
    return sessions, normalized


# --------------------------------------------------------------------------- #
# 刷卡窗口、强制在场与结构性矛盾
# --------------------------------------------------------------------------- #
def clipped_scan(scan, session):
    """把刷卡的进、出闭区间裁剪到课程窗口，返回 (enter_lo, enter_hi, leave_lo, leave_hi)。

    离场允许取课程 end（即最后一分钟结束时离开），入场最晚取 end-1。
    """
    enter_lo = max(scan["enter_lo"], session["start"])
    enter_hi = min(scan["enter_hi"], session["end"] - 1)
    leave_lo = max(scan["leave_lo"], session["start"])
    leave_hi = min(scan["leave_hi"], session["end"])
    return enter_lo, enter_hi, leave_lo, leave_hi


def forced_minutes(scan, session):
    """该刷卡记录在任意可行选择下都必然在场的分钟集合（[enter_hi, leave_lo)）。"""
    enter_lo, enter_hi, leave_lo, leave_hi = clipped_scan(scan, session)
    if enter_lo > enter_hi or leave_lo > leave_hi or leave_hi < enter_lo:
        return set()
    return set(range(enter_hi, leave_lo))


def structural_diagnostics(tables, sessions):
    """不依赖求解器的矛盾定位：窗口冲突与同学员区间不兼容。

    返回矛盾列表；为空表示未发现结构性矛盾，整体可行性仍需求解模型确认。
    """
    session_rows = {row["session"]: row for row in sessions}
    scans_by_learner = defaultdict(list)
    contradictions = []
    for scan in tables["scans"]:
        session = session_rows[scan["session"]]
        enter_lo, enter_hi, leave_lo, leave_hi = clipped_scan(scan, session)
        if enter_lo > enter_hi or leave_lo > leave_hi or leave_hi < enter_lo:
            contradictions.append({
                "learner": scan["learner"], "session": scan["session"],
                "reason": "刷卡时间裁剪到课程窗口后无可行入场或离场时刻",
            })
            continue
        scans_by_learner[scan["learner"]].append(
            (scan["session"], session["day"], enter_lo, enter_hi, leave_lo, leave_hi))
    for learner, items in scans_by_learner.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sid_a, day_a, elo_a, ehi_a, llo_a, lhi_a = items[i]
                sid_b, day_b, elo_b, ehi_b, llo_b, lhi_b = items[j]
                if day_a != day_b:
                    continue
                # 两个到课区间 [e,l) 可先后安排的条件：
                # A 先于 B 要求 llo_A <= ehi_B；B 先于 A 要求 llo_B <= ehi_A。
                if not (llo_a <= ehi_b or llo_b <= ehi_a):
                    contradictions.append({
                        "learner": learner,
                        "session": sorted([sid_a, sid_b]),
                        "reason": "同一学员在重叠课程的刷卡区间无法先后安排，必然同时出现在两个教室",
                    })
    contradictions.sort(key=lambda item: (str(item["session"]), str(item["learner"])))
    return contradictions


def capacity_diagnostics(sessions, evidence):
    """基于强制在场分钟定位必然超容的教室分钟（无需求解器）。

    每条刷卡在 [enter_hi, leave_lo) 必然在场，若某教室某分钟的必然人数超过
    当时生效容量，则任何安排都无解，可直接定位到相互矛盾的记录集合。
    """
    forced_learners = defaultdict(set)
    for (learner, day), items in evidence.forced.items():
        for minute, room in items:
            forced_learners[(room, day, minute)].add(learner)
    # 先收集每个 (room,day) 的全部超容分钟，再聚合成一条诊断，避免逐分钟刷屏。
    overloaded = defaultdict(list)
    for session_row in sessions:
        for minute in range(session_row["start"], session_row["end"]):
            room, day = session_row["room"], session_row["day"]
            active = [row for row in sessions
                      if row["room"] == room and row["day"] == day
                      and row["start"] <= minute < row["end"]]
            capacity = min(row["capacity"] for row in active)
            learners = forced_learners.get((room, day, minute), set())
            if len(learners) > capacity:
                overloaded[(room, day)].append({
                    "minute": minute, "forced": len(learners), "capacity": capacity,
                    "sessions": sorted(row["session"] for row in active),
                    "learners": sorted(learners),
                })
    contradictions = []
    for (room, day), entries in sorted(overloaded.items()):
        worst = max(entries, key=lambda item: item["forced"] - item["capacity"])
        involved = sorted({learner for entry in entries for learner in entry["learners"]})
        contradictions.append({
            "learner": involved,
            "session": sorted({sid for entry in entries for sid in entry["sessions"]}),
            "reason": (f"教室 {room} 第 {day} 天有 {len(entries)} 个分钟被刷卡记录强制超容；"
                       f"最严重为分钟 {worst['minute']}，{worst['forced']} 人超过生效容量 "
                       f"{worst['capacity']}，涉及 {len(involved)} 名学员"),
            "room": room, "day": day,
            "overloaded_minutes": [entry["minute"] for entry in entries],
            "worst_minute": worst["minute"], "worst_forced": worst["forced"],
            "capacity": worst["capacity"],
        })
    return contradictions


class EvidenceIndex:
    """按学员/会话索引刷卡窗口与强制在场分钟，供组合边界与证据留存复用。"""

    def __init__(self, tables, sessions):
        self.session_rows = {row["session"]: row for row in sessions}
        self.scan_map = {(row["learner"], row["session"]): row for row in tables["scans"]}
        self.window = {}          # (learner, session) -> (enter_lo, leave_hi)
        self.forced = defaultdict(set)  # (learner, day) -> {(minute, room)}
        self.forced_evidence = []
        for scan in sorted(tables["scans"], key=lambda row: (row["session"], row["learner"])):
            session = self.session_rows[scan["session"]]
            enter_lo, enter_hi, leave_lo, leave_hi = clipped_scan(scan, session)
            self.window[(scan["learner"], scan["session"])] = (enter_lo, leave_hi)
            minutes = forced_minutes(scan, session)
            self.forced_evidence.append({
                "learner": scan["learner"], "session": scan["session"],
                "forced_minutes": sorted(minutes),
            })
            for minute in minutes:
                self.forced[(scan["learner"], session["day"])].add((minute, session["room"]))
        self.bookings_by_session = defaultdict(list)
        for booking in tables["bookings"]:
            self.bookings_by_session[booking["session"]].append(booking["learner"])

    def can_attend_minute(self, learner, session_id, room, day, minute):
        """学员在 minute 是否可能出现在该会话所在教室。"""
        scan = self.scan_map.get((learner, session_id))
        if scan is not None:
            enter_lo, leave_hi = self.window[(learner, session_id)]
            if not (enter_lo <= minute < leave_hi):
                return False
        # 被另一间教室的强制在场分钟钉住时不可能在此。
        if any(other_minute == minute and other_room != room
               for other_minute, other_room in self.forced.get((learner, day), ())):
            return False
        return True


# --------------------------------------------------------------------------- #
# 组合精确边界：人头与同时占用峰值
# --------------------------------------------------------------------------- #
def headcount_bounds(session_row, evidence, scan_counts):
    """单会话到课人头的精确上下界。

    下界为刷卡人数（必然到课）；上界为下界加上“至少能在课程窗口某分钟到场”
    的未刷卡报名者。被其他教室全程钉住的未刷卡者无法到课。
    """
    session_id = session_row["session"]
    lower = scan_counts[session_id]
    upper = lower
    blocked_learners = []
    for learner in evidence.bookings_by_session[session_id]:
        if (learner, session_id) in evidence.scan_map:
            continue
        blocked = {minute for minute, room in evidence.forced.get((learner, session_row["day"]), ())
                   if room != session_row["room"]}
        if any(minute not in blocked
               for minute in range(session_row["start"], session_row["end"])):
            upper += 1
        else:
            blocked_learners.append(learner)
    return lower, upper, sorted(blocked_learners)


def peak_bounds(segment, evidence, forced_count):
    """房间分段内同时占用的精确上下界及对应分钟。

    下界取强制在场分钟的最大同时占用；上界逐分钟枚举“可能到场学员数”，
    再与该分钟生效容量取最小。
    """
    lower = 0
    lower_minute = None
    for minute in range(segment["start"], segment["end"]):
        value = forced_count[(segment["room"], segment["day"], minute)]
        if value > lower:
            lower, lower_minute = value, minute
    upper = 0
    upper_minute = None
    for minute in range(segment["start"], segment["end"]):
        capacity = segment["minute_capacity"][minute]
        eligible = set()
        for session_id in segment["sessions"]:
            session_row = evidence.session_rows[session_id]
            if not (session_row["start"] <= minute < session_row["end"]):
                continue
            for learner in evidence.bookings_by_session[session_id]:
                if evidence.can_attend_minute(
                        learner, session_id, segment["room"], segment["day"], minute):
                    eligible.add(learner)
        value = min(capacity, len(eligible))
        if value > upper:
            upper, upper_minute = value, minute
    return lower, lower_minute, upper, upper_minute


def build_segments(sessions):
    """按 room/day 合并相重叠或相邻的课程窗口，得到对外汇报的时间分段。"""
    grouped = defaultdict(list)
    for row in sessions:
        grouped[(row["day"], row["room"])].append((row["start"], row["end"], row["session"]))
    segments = []
    for (day, room), items in sorted(grouped.items()):
        items.sort()
        current_start = current_end = None
        members = []
        for start, end, session_id in items:
            if current_start is None:
                current_start, current_end = start, end
                members = [session_id]
            elif start <= current_end:
                current_end = max(current_end, end)
                members.append(session_id)
            else:
                segments.append({"day": day, "room": room, "start": current_start,
                                 "end": current_end, "sessions": sorted(members)})
                current_start, current_end = start, end
                members = [session_id]
        if current_start is not None:
            segments.append({"day": day, "room": room, "start": current_start,
                             "end": current_end, "sessions": sorted(members)})
    segments.sort(key=lambda item: (item["day"], item["room"], item["start"]))
    # 每分钟生效容量 = 当时占用该房间的会话容量的最小值。
    for segment in segments:
        minute_capacity = {}
        for minute in range(segment["start"], segment["end"]):
            active = [evidence_row for evidence_row in sessions
                      if evidence_row["room"] == segment["room"]
                      and evidence_row["day"] == segment["day"]
                      and evidence_row["start"] <= minute < evidence_row["end"]]
            minute_capacity[minute] = min(row["capacity"] for row in active)
        segment["minute_capacity"] = minute_capacity
        segment["capacity"] = min(minute_capacity.values())
    return segments


# --------------------------------------------------------------------------- #
# 座位分钟：分钟级流守恒线性规划
# --------------------------------------------------------------------------- #
class DayModel:
    """单日的分钟级占用模型，仅用于座位分钟总量边界。

    每条报名占用 3*n 个 [0,1] 变量：在场 x_t、入场 y_t、离场 z_t。
    刷卡报名用 y/z 的和为 1 与允许时刻约束表达；未刷卡报名允许全零（缺席）。
    流守恒 x_t = x_{t-1} + y_t - z_t 保证在场是单个连续区间（或全程缺席）。
    """

    def __init__(self, day, tables, sessions):
        self.day = day
        self.session_rows = {row["session"]: row for row in sessions if row["day"] == day}
        self.scan_map = {(row["learner"], row["session"]): row for row in tables["scans"]}
        self.blocks = []
        self._build(tables)

    def _build(self, tables):
        rows, cols, vals, lower, upper = [], [], [], [], []
        integrality = []
        column = 0

        def add_row(coefs, lo, hi):
            row_index = len(lower)
            lower.append(lo)
            upper.append(hi)
            for col_index, value in coefs:
                rows.append(row_index)
                cols.append(col_index)
                vals.append(value)

        day_bookings = [row for row in tables["bookings"] if row["session"] in self.session_rows]
        day_bookings.sort(key=lambda row: (row["session"], row["learner"]))
        for booking in day_bookings:
            session = self.session_rows[booking["session"]]
            start, end = session["start"], session["end"]
            n = end - start
            x0, y0, z0 = column, column + n, column + 2 * n
            column += 3 * n
            integrality.extend([0] * n)          # x 先松弛；回退时整体整化
            integrality.extend([1, 1] * n)       # y、z 为二进制
            x_index = lambda t, x0=x0: x0 + (t - start)
            y_index = lambda t, y0=y0: y0 + (t - start)
            z_index = lambda t, z0=z0: z0 + (t - start)
            for t in range(start, end):
                coef = [(x_index(t), 1.0), (y_index(t), -1.0), (z_index(t), 1.0)]
                if t > start:
                    coef.append((x_index(t - 1), -1.0))
                add_row(coef, 0.0, 0.0)
            # 课程结束时不得仍然在场：最后一分钟的在场必须伴随该分钟离场。
            add_row([(x_index(end - 1), 1.0), (z_index(end - 1), -1.0)], -np.inf, 0.0)
            scan = self.scan_map.get((booking["learner"], booking["session"]))
            if scan is None:
                add_row([(y_index(t), 1.0) for t in range(start, end)], 0.0, 1.0)
                add_row([(z_index(t), 1.0) for t in range(start, end)], 0.0, 1.0)
            else:
                enter_lo, enter_hi, leave_lo, leave_hi = clipped_scan(scan, session)
                add_row([(y_index(t), 1.0) for t in range(start, end)], 1.0, 1.0)
                add_row([(z_index(t), 1.0) for t in range(start, end)], 1.0, 1.0)
                for t in range(start, end):
                    if not (enter_lo <= t <= enter_hi):
                        add_row([(y_index(t), 1.0)], 0.0, 0.0)
                    if not (leave_lo <= t <= leave_hi):
                        add_row([(z_index(t), 1.0)], 0.0, 0.0)
                # 入场不晚于离场：sum(t*y_t) - sum(t*z_t) <= 0。
                add_row(
                    [(y_index(t), float(t)) for t in range(start, end)]
                    + [(z_index(t), float(-t)) for t in range(start, end)],
                    -np.inf, 0.0,
                )
            self.blocks.append({
                "learner": booking["learner"], "session": booking["session"],
                "room": session["room"], "start": start, "end": end,
                "capacity": session["capacity"], "x0": x0, "n": n,
                "scanned": scan is not None,
            })

        learner_minute = defaultdict(list)
        room_minute = defaultdict(list)
        for block in self.blocks:
            for t in range(block["start"], block["end"]):
                x_variable = block["x0"] + (t - block["start"])
                learner_minute[(block["learner"], t)].append((x_variable, 1.0))
                room_minute[(block["room"], t)].append((x_variable, 1.0))
        for coef in learner_minute.values():
            if len(coef) > 1:
                add_row(coef, -np.inf, 1.0)
        for (room, minute), coef in room_minute.items():
            active = [row for row in self.session_rows.values()
                      if row["room"] == room and row["start"] <= minute < row["end"]]
            capacity = min(row["capacity"] for row in active)
            add_row(coef, -np.inf, float(capacity))

        self.nvar = column
        self.matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), column)).tocsr()
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.integrality = np.array(integrality)

    @staticmethod
    def _status(result):
        if result.status == 0:
            return "optimal"
        return "infeasible" if result.status == 2 else "unknown"

    def feasibility(self):
        """整体可行性检查；松弛解为整数即确认存在可行整数安排，否则回退 MIP。"""
        zero_cost = np.zeros(self.nvar)
        result = milp(c=zero_cost, constraints=LinearConstraint(self.matrix, self.lower, self.upper),
                      integrality=self.integrality, bounds=Bounds(0, 1),
                      options={"time_limit": SOLVER_TIME_LIMIT, "disp": False})
        status = self._status(result)
        if status != "optimal":
            return status
        if np.all(np.abs(result.x - np.round(result.x)) <= INTEGRAL_TOLERANCE):
            return "optimal"
        result = milp(c=zero_cost, constraints=LinearConstraint(self.matrix, self.lower, self.upper),
                      integrality=np.ones(self.nvar), bounds=Bounds(0, 1),
                      options={"time_limit": SOLVER_TIME_LIMIT, "disp": False})
        return self._status(result)

    def seat_minutes(self, session_id, sense):
        """会话座位分钟总量的 min/max，返回 (status, value, used_integer_fallback)。"""
        cost = np.zeros(self.nvar)
        for block in self.blocks:
            if block["session"] == session_id:
                cost[block["x0"]:block["x0"] + block["n"]] = 1.0
        sign = 1.0 if sense == "min" else -1.0
        result = milp(c=sign * cost, constraints=LinearConstraint(self.matrix, self.lower, self.upper),
                      integrality=self.integrality, bounds=Bounds(0, 1),
                      options={"time_limit": SOLVER_TIME_LIMIT, "disp": False})
        status = self._status(result)
        if status != "optimal":
            return status, None, False
        integer_fallback = False
        if np.any(np.abs(result.x - np.round(result.x)) > INTEGRAL_TOLERANCE):
            integer_fallback = True
            result = milp(c=sign * cost,
                          constraints=LinearConstraint(self.matrix, self.lower, self.upper),
                          integrality=np.ones(self.nvar), bounds=Bounds(0, 1),
                          options={"time_limit": SOLVER_TIME_LIMIT, "disp": False})
            status = self._status(result)
            if status != "optimal":
                return status, None, integer_fallback
        return "optimal", int(round(sign * result.fun)), integer_fallback


# --------------------------------------------------------------------------- #
# 压力分级与结果组装
# --------------------------------------------------------------------------- #
def _counts_by_session(tables, table_name):
    counts = defaultdict(int)
    for row in tables[table_name]:
        counts[row["session"]] += 1
    return counts


def _ratio(value, denominator):
    if value is None or not denominator:
        return None
    return round(value / denominator, 6)


def _headcount_pressure(lower, upper, capacity, booked):
    if lower >= capacity:
        return "confirmed_over_capacity"
    if upper is not None and upper >= capacity:
        return "possible_over_capacity"
    if upper == 0 and booked > 0:
        return "evidence_gap"
    return "within_capacity"


def _segment_pressure(lower, upper, capacity):
    if lower >= capacity:
        return "confirmed_over_capacity"
    if upper is None:
        return "unknown"
    if upper >= capacity:
        return "possible_over_capacity"
    if lower == 0:
        return "evidence_gap_or_idle"
    return "within_capacity"


def analyze_scenario(tables, sessions, label):
    """对一套 sessions 配置计算完整占用边界。"""
    sessions = sorted(sessions, key=lambda row: row["session"])
    days = sorted({row["day"] for row in sessions})
    contradictions = structural_diagnostics(tables, sessions)
    evidence = EvidenceIndex(tables, sessions)
    contradictions.extend(capacity_diagnostics(sessions, evidence))
    scan_counts = _counts_by_session(tables, "scans")
    booking_counts = _counts_by_session(tables, "bookings")
    forced_count = defaultdict(int)
    for (learner, day), items in evidence.forced.items():
        for minute, room in items:
            forced_count[(room, day, minute)] += 1

    solver_status = "not_needed"
    seat_minutes_fallback = False
    session_results = []
    segment_results = []
    if not contradictions:
        models = {day: DayModel(day, tables, sessions) for day in days}
        for day, model in models.items():
            status = model.feasibility()
            if status != "optimal":
                solver_status = status
                contradictions.append({
                    "learner": None, "session": None,
                    "reason": f"第 {day} 天的约束模型无解或求解器无法确认可行（状态：{status}），需人工复核容量与刷卡记录",
                })
                break
        if not contradictions:
            solver_status = "optimal"
            for session_row in sessions:
                model = models[session_row["day"]]
                session_id = session_row["session"]
                lower_head, upper_head, blocked = headcount_bounds(
                    session_row, evidence, scan_counts)
                min_status, min_seats, fallback_min = model.seat_minutes(session_id, "min")
                max_status, max_seats, fallback_max = model.seat_minutes(session_id, "max")
                seat_minutes_fallback = seat_minutes_fallback or fallback_min or fallback_max
                window = session_row["end"] - session_row["start"]
                capacity_seat_minutes = session_row["capacity"] * window
                session_results.append({
                    "session": session_id,
                    "day": session_row["day"],
                    "room": session_row["room"],
                    "start": session_row["start"],
                    "end": session_row["end"],
                    "capacity": session_row["capacity"],
                    "booked": booking_counts[session_id],
                    "scan_observed": scan_counts[session_id],
                    "headcount_lower": lower_head,
                    "headcount_upper": upper_head,
                    "headcount_blocked_learners": blocked,
                    "seat_minutes_lower": min_seats if min_status == "optimal" else None,
                    "seat_minutes_upper": max_seats if max_status == "optimal" else None,
                    "seat_minutes_window": window,
                    "seat_minutes_capacity": capacity_seat_minutes,
                    "utilization_lower": _ratio(
                        min_seats if min_status == "optimal" else None, capacity_seat_minutes),
                    "utilization_upper": _ratio(
                        max_seats if max_status == "optimal" else None, capacity_seat_minutes),
                    "headcount_pressure": _headcount_pressure(
                        lower_head, upper_head, session_row["capacity"],
                        booking_counts[session_id]),
                    "bound_basis": "人头为刷卡强制到课与逐分钟可行性的组合判定；座位分钟为分钟级流守恒线性规划",
                })
            for segment in build_segments(sessions):
                lower_peak, lower_minute, upper_peak, upper_minute = peak_bounds(
                    segment, evidence, forced_count)
                segment_results.append({
                    "day": segment["day"], "room": segment["room"],
                    "start": segment["start"], "end": segment["end"],
                    "sessions": segment["sessions"],
                    "peak_lower": lower_peak,
                    "peak_lower_minute": lower_minute,
                    "peak_upper": upper_peak,
                    "peak_upper_minute": upper_minute,
                    "capacity": segment["capacity"],
                    "pressure": _segment_pressure(lower_peak, upper_peak, segment["capacity"]),
                    "bound_basis": "下界取强制在场分钟的最大同时占用；上界逐分钟枚举可能到场学员并受容量约束",
                })
    else:
        solver_status = "infeasible"

    return {
        "scenario": label,
        "feasible": not contradictions,
        "solver_status": solver_status,
        "seat_minutes_integer_fallback": seat_minutes_fallback,
        "sessions": session_results,
        "segments": segment_results,
        "contradictions": contradictions,
        "evidence": {
            "forced_presence": evidence.forced_evidence,
            "scan_records": len(tables["scans"]),
            "booking_records": len(tables["bookings"]),
        },
    }


def occupancy_bounds(tables, parameters):
    """ANALYSES 登记入口：计算基线与可选场地情景并比较变化。"""
    normalized = normalize_parameters(parameters)
    baseline = analyze_scenario(tables, [dict(row) for row in tables["sessions"]], "baseline")
    result = {
        "analysis": "occupancy_bounds",
        "parameters": normalized,
        "assumptions": ASSUMPTIONS,
        "baseline": baseline,
    }
    if normalized["scenario"]:
        scenario_sessions, _ = apply_scenario(tables, normalized)
        scenario = analyze_scenario(tables, scenario_sessions, "scenario")
        result["scenario"] = scenario
        result["comparison"] = compare_scenarios(baseline, scenario)
    return result


def compare_scenarios(baseline, scenario):
    """按 session 对齐两套配置的边界，给出座位时长与利用率的变化。"""
    if not baseline["feasible"] or not scenario["feasible"]:
        return {
            "note": "存在无解配置，仅比较可行性；边界变化不做数值汇总。",
            "baseline_feasible": baseline["feasible"],
            "scenario_feasible": scenario["feasible"],
            "sessions": [],
        }
    base = {row["session"]: row for row in baseline["sessions"]}
    changed = {row["session"]: row for row in scenario["sessions"]}
    sessions = []
    for session_id in sorted(set(base) | set(changed)):
        left, right = base.get(session_id), changed.get(session_id)
        if left is None or right is None:
            continue
        sessions.append({
            "session": session_id,
            "room_changed": left["room"] != right["room"],
            "baseline_room": left["room"],
            "scenario_room": right["room"],
            "capacity_change": right["capacity"] - left["capacity"],
            "headcount_upper_change": _delta(right["headcount_upper"], left["headcount_upper"]),
            "seat_minutes_upper_change": _delta(right["seat_minutes_upper"], left["seat_minutes_upper"]),
            "seat_minutes_lower_change": _delta(right["seat_minutes_lower"], left["seat_minutes_lower"]),
            "utilization_upper_change": _ratio_delta(right["utilization_upper"], left["utilization_upper"]),
        })
    return {
        "note": "各变化量为情景值减基线值；极值仍分别求解，不代表同一到课安排。",
        "baseline_feasible": baseline["feasible"],
        "scenario_feasible": scenario["feasible"],
        "sessions": sessions,
    }


def _delta(after, before):
    if after is None or before is None:
        return None
    return after - before


def _ratio_delta(after, before):
    if after is None or before is None:
        return None
    return round(after - before, 6)
