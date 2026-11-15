from collections import Counter, defaultdict
import math

import numpy as np

from learning.codec import numeric, require_fields, unique
from learning.occupancy import occupancy_bounds

PRODUCT = '课室使用分析'
KEYS = {'sessions': ['session'], 'bookings': ['learner', 'session'], 'scans': ['learner', 'session']}


def validate(tables):
    if set(tables) != set(KEYS):
        raise ValueError("资料表集合与产品契约不一致")
    for name, fields in KEYS.items():
        require_fields(tables[name], fields)
        unique(tables[name], fields)
    session_ids = {row["session"] for row in tables["sessions"]}
    bookings = {(row["learner"], row["session"]) for row in tables["bookings"]}
    for row in tables["sessions"]:
        if not row["start"] < row["end"] or row["capacity"] <= 0:
            raise ValueError("课时或容量不合法")
    for row in tables["bookings"]:
        if row["session"] not in session_ids:
            raise ValueError("报名课程不存在")
    for row in tables["scans"]:
        if (row["learner"], row["session"]) not in bookings:
            raise ValueError("刷卡记录没有报名关系")
        if row["enter_lo"] > row["enter_hi"] or row["leave_lo"] > row["leave_hi"]:
            raise ValueError("刷卡区间方向不合法")


def overview(tables, parameters):
    booked = Counter(row["session"] for row in tables["bookings"])
    observed = Counter(row["session"] for row in tables["scans"])
    return {"courses": [{**row, "booked": booked[row["session"]], "scanned": observed[row["session"]]}
                        for row in sorted(tables["sessions"], key=lambda row: row["session"])], "learners": len({row["learner"] for row in tables["bookings"]})}


ANALYSES = {"overview": overview, "occupancy_bounds": occupancy_bounds}
