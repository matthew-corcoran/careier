"""REBA-style scoring for a live care-worker lift coach.

Tables follow Hignett & McAtamney, Applied Ergonomics 31 (2000).
Webcam pose cannot see wrist flexion or true load, so those extras are
fixed for patient handling and the on-screen label says "REBA-style".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# table_a[trunk-1][neck-1][legs-1]
TABLE_A = [
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]

# table_b[upper_arm-1][lower_arm-1][wrist-1]
TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]],
    [[6, 7, 8], [7, 8, 8]],
    [[7, 8, 8], [8, 9, 9]],
]

# table_c[score_a-1][score_b-1]  (scores 1..12)
TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]

# Camera demo defaults. A real REBA assessment would add more for patient
# weight and an awkward grip, but those two flat penalties made a careful
# kneel look the same as a stoop. One point keeps a person harder than a box.
LOAD_SCORE = 1
COUPLING_SCORE = 1
DEFAULT_WRIST_SCORE = 2

RISK_BANDS = (
    (1, 1, "NEGLIGIBLE"),
    (2, 3, "LOW"),
    (4, 7, "MEDIUM"),
    (8, 10, "HIGH"),
    (11, 15, "VERY HIGH"),
)


@dataclass(frozen=True)
class RebaAngles:
    trunk_flex_deg: float
    trunk_twist_deg: float
    trunk_side_deg: float
    neck_flex_deg: float
    neck_twist: bool
    knee_flex_deg: float
    upper_arm_deg: float
    lower_arm_deg: float
    arm_abducted: bool = False
    unstable_base: bool = False


@dataclass
class RebaResult:
    score: int
    band: str
    trunk: int
    neck: int
    legs: int
    upper_arm: int
    lower_arm: int
    wrist: int
    table_a: int
    table_b: int
    score_a: int
    score_b: int
    table_c: int
    load: int = LOAD_SCORE
    coupling: int = COUPLING_SCORE
    activity: int = 0
    parts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parts:
            self.parts = {
                "Trunk": self.trunk,
                "Neck": self.neck,
                "Legs": self.legs,
                "Arm": self.upper_arm,
            }


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def risk_band(score: int) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= score <= hi:
            return name
    return "VERY HIGH"


def _lookup(table: list, *indices: int) -> int:
    cell = table
    for index in indices:
        cell = cell[index]
    return int(cell)


def trunk_score(flex_deg: float, twist_deg: float, side_deg: float) -> int:
    flex = abs(flex_deg)
    if flex < 5:
        base = 1
    elif flex <= 20:
        base = 2
    elif flex <= 60:
        base = 3
    else:
        base = 4
    # 2D pose reports small twist and side bend on almost every lean.
    if abs(twist_deg) >= 25 or abs(side_deg) >= 25:
        base += 1
    return clamp(base, 1, 5)


def neck_score(flex_deg: float, twisted: bool) -> int:
    # Extension (negative) or large flexion is worse.
    if flex_deg < -5 or flex_deg > 20:
        base = 2
    else:
        base = 1
    if twisted:
        base += 1
    return clamp(base, 1, 3)


def legs_score(knee_flex_deg: float, unstable_base: bool) -> int:
    # A deep kneel is the safe way to get someone off the floor. Textbook REBA
    # adds two points past 60°, and the camera then saturates a tucked shin
    # near 180°. Count a partial squat, not a supported kneel.
    base = 2 if unstable_base else 1
    knee = abs(knee_flex_deg)
    if 30 <= knee <= 110:
        base += 1
    return clamp(base, 1, 4)


def upper_arm_score(elev_deg: float, abducted: bool) -> int:
    elev = abs(elev_deg)
    # Shifted up from the paper's 20/45/90 bins. A side camera overstates
    # how high the arm is while the carer reaches around someone.
    if elev <= 30:
        base = 1
    elif elev <= 60:
        base = 2
    elif elev <= 100:
        base = 3
    else:
        base = 4
    if abducted:
        base += 1
    return clamp(base, 1, 6)


def lower_arm_score(elbow_deg: float) -> int:
    # REBA: 60–100° flexion is the better bin.
    if 60 <= elbow_deg <= 100:
        return 1
    return 2


def score_reba(angles: RebaAngles, activity: int = 0) -> RebaResult:
    trunk = trunk_score(angles.trunk_flex_deg, angles.trunk_twist_deg, angles.trunk_side_deg)
    neck = neck_score(angles.neck_flex_deg, angles.neck_twist)
    legs = legs_score(angles.knee_flex_deg, angles.unstable_base)
    upper = upper_arm_score(angles.upper_arm_deg, angles.arm_abducted)
    lower = lower_arm_score(angles.lower_arm_deg)
    wrist = DEFAULT_WRIST_SCORE

    table_a = _lookup(TABLE_A, trunk - 1, neck - 1, legs - 1)
    table_b = _lookup(TABLE_B, upper - 1, lower - 1, wrist - 1)
    score_a = clamp(table_a + LOAD_SCORE, 1, 12)
    score_b = clamp(table_b + COUPLING_SCORE, 1, 12)
    table_c = _lookup(TABLE_C, score_a - 1, score_b - 1)
    activity = clamp(int(activity), 0, 3)
    final = clamp(table_c + activity, 1, 15)

    return RebaResult(
        score=final,
        band=risk_band(final),
        trunk=trunk,
        neck=neck,
        legs=legs,
        upper_arm=upper,
        lower_arm=lower,
        wrist=wrist,
        table_a=table_a,
        table_b=table_b,
        score_a=score_a,
        score_b=score_b,
        table_c=table_c,
        activity=activity,
    )


# Tuned demo postures: stoop+twist must be 8+, squat-close under 7.
STOOP_TWIST = RebaAngles(
    trunk_flex_deg=72,
    trunk_twist_deg=28,
    trunk_side_deg=18,
    neck_flex_deg=32,
    neck_twist=True,
    knee_flex_deg=8,
    upper_arm_deg=95,
    lower_arm_deg=150,
    arm_abducted=True,
    unstable_base=True,
)

SQUAT_CLOSE = RebaAngles(
    trunk_flex_deg=18,
    trunk_twist_deg=6,
    trunk_side_deg=4,
    neck_flex_deg=12,
    neck_twist=False,
    knee_flex_deg=42,
    upper_arm_deg=32,
    lower_arm_deg=85,
    arm_abducted=False,
    unstable_base=False,
)


def _check_demo_thresholds() -> None:
    bad = score_reba(STOOP_TWIST, activity=1)
    good = score_reba(SQUAT_CLOSE, activity=0)
    if bad.score < 8:
        raise SystemExit(f"stoop-twist REBA {bad.score} expected >= 8")
    if good.score >= 7:
        raise SystemExit(f"squat-close REBA {good.score} expected < 7")
    print(f"stoop-twist  REBA {bad.score} {bad.band}  (A={bad.score_a} B={bad.score_b})")
    print(f"squat-close  REBA {good.score} {good.band}  (A={good.score_a} B={good.score_b})")


if __name__ == "__main__":
    _check_demo_thresholds()
