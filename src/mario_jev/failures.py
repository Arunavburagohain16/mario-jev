"""Death lessons from finished runs, replayed into later observations.

This is not model training. A terminated episode becomes a short record of where
Mario died, which jump led there, and which buttons were held. Later decisions
include that record from before the fatal takeoff. A repeated high jump is cut
short. The fatal button is replaced near the death only when an enemy is not
already touching Mario.
"""

import json
from collections import defaultdict, deque
from pathlib import Path

APPROACH_PX = 192
REPEAT_PX = 48
HIGH_FALL_PX = 32
COMMIT_LEAD_PX = 80
LAUNCH_AFTER_COMMIT_PX = 96
SHORT_JUMP_PX = 64
HEIGHT_CAP_PX = 32
RECENT_DECISIONS = 48

WITHOUT_JUMP = {
    "right_run_jump": "right_run",
    "right_jump": "right",
    "left_jump": "left",
    "jump": "wait",
}

ALTERNATIVES = {
    "right_run_jump": ("right_run", "right_jump", "right"),
    "right_run": ("right_run_jump", "right", "right_jump"),
    "right_jump": ("right_run_jump", "right", "right_run"),
    "right": ("right_run", "right_jump", "right_run_jump"),
    "left_jump": ("left", "right_run_jump", "jump"),
    "left": ("left_jump", "wait", "right"),
    "jump": ("wait", "right_jump", "right_run_jump"),
    "wait": ("jump", "right", "right_run"),
}


def hazard_from_state(state):
    objects = [
        obj
        for obj in state.get("nearby_objects") or []
        if obj.get("kind") != "flagpole" and not obj.get("defeated")
    ]
    if objects:
        nearest = min(objects, key=lambda obj: abs(obj.get("dx", 0)))
        parts = [f"{nearest.get('kind', 'object')} dx={nearest.get('dx')}"]
        if "dy" in nearest:
            parts.append(f"dy={nearest['dy']}")
        if nearest.get("estimated_contact_in_frames") is not None:
            parts.append(f"contact_frames={nearest['estimated_contact_in_frames']}")
        return ", ".join(parts)
    summary = (state.get("terrain") or {}).get("summary") or {}
    gap = summary.get("nearest_empty_column_below_feet")
    if isinstance(gap, dict) and gap.get("edge_distance_px") is not None:
        return f"missing floor {gap['edge_distance_px']}px ahead"
    mario = state.get("mario") or {}
    if mario.get("motion") == "falling":
        return "falling with no enemy in view"
    return "death with no enemy or pit in the observation"


def lesson_from_decision(decision, actions):
    result = decision.get("result") or {}
    state = decision.get("state") or {}
    choices = decision.get("decisions") or {}
    mario = state.get("mario") or {}
    death_x = int(result.get("x", mario.get("x", 0)))
    return {
        "death_x": death_x,
        "death_y": int(result.get("y", mario.get("y", 0))),
        "fatal_action": decision.get("action"),
        "movement": choices.get("movement"),
        "jump_pressed": choices.get("jump_pressed"),
        "actions_before_death": list(actions)[-8:],
        "hazard": hazard_from_state(state),
        "note": (
            "A previous run died here with fatal_action. "
            "Do not repeat that action or the same approach."
        ),
    }


def _mario(decision):
    return (decision.get("state") or {}).get("mario") or {}


def _launched(decision):
    mario = _mario(decision)
    state = decision.get("state") or {}
    return bool(
        mario.get("grounded")
        and not state.get("jump_already_held")
        and "jump" in (decision.get("action") or "")
    )


def _survived_jump(decisions, index):
    streak = 0
    for later in decisions[index + 1 : -1]:
        if _mario(later).get("grounded"):
            streak += 1
            if streak >= 3:
                return True
        else:
            streak = 0
    return False


def find_commit(decisions):
    """Latest high jump that was never followed by a safe landing before death."""
    death = decisions[-1]
    mario = _mario(death)
    result = death.get("result") or {}
    fallback_y = int(result.get("y", mario.get("y", 0)))
    fallback = {
        "commit_x": int(result.get("x", mario.get("x", 0))),
        "commit_y": fallback_y,
        "commit_action": death.get("action"),
        "peak_y": fallback_y,
        "fall_height_px": 0,
    }
    chosen = None
    for index, decision in enumerate(decisions[:-1]):
        if not _launched(decision) or _survived_jump(decisions, index):
            continue
        takeoff = _mario(decision).get("y")
        takeoff_x = _mario(decision).get("x")
        if takeoff is None or takeoff_x is None:
            continue
        peak = takeoff
        for later in decisions[index:]:
            for value in (_mario(later).get("y"), (later.get("result") or {}).get("y")):
                if value is not None:
                    peak = min(peak, value)
        fall = int(takeoff) - int(peak)
        if fall < HIGH_FALL_PX:
            continue
        chosen = {
            "commit_x": int(takeoff_x),
            "commit_y": int(takeoff),
            "commit_action": decision.get("action"),
            "peak_y": int(peak),
            "fall_height_px": fall,
        }
    return chosen or fallback


def lesson_from_episode(decisions):
    lesson = lesson_from_decision(
        decisions[-1], [decision.get("action") for decision in decisions]
    )
    lesson.update(find_commit(decisions))
    if lesson["fall_height_px"] >= HIGH_FALL_PX:
        lesson["note"] = (
            "A previous run jumped at commit_x and climbed fall_height_px, "
            "then fell into the hazard at death_x. Do not start or hold that jump."
        )
    return lesson


def collapse(lessons):
    merged = {}
    order = []
    for lesson in lessons:
        key = (int(lesson["death_x"]) // 32, lesson["fatal_action"])
        seen = int(lesson.get("times_seen", 1))
        if key not in merged:
            order.append(key)
            merged[key] = {**lesson, "times_seen": seen}
            continue
        merged[key] = {**lesson, "times_seen": merged[key]["times_seen"] + seen}
    return [merged[key] for key in order]


def lessons_in_log(path, env_id=None):
    recent = defaultdict(lambda: deque(maxlen=RECENT_DECISIONS))
    lessons = []
    with Path(path).open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            episode = record.get("episode", 0)
            kind = record.get("type")
            if kind == "config" and env_id is not None and record.get("env") != env_id:
                return []
            if kind == "decision" and record.get("action"):
                recent[episode].append(record)
            elif (
                kind == "summary"
                and record.get("stop_reason") == "terminated"
                and not record.get("completed")
                and recent.get(episode)
            ):
                lessons.append(lesson_from_episode(list(recent[episode])))
    return lessons


def already_touching(state):
    if not state:
        return False
    for obj in state.get("nearby_objects") or []:
        if obj.get("kind") == "flagpole" or obj.get("defeated"):
            continue
        dx = obj.get("dx")
        if dx is None or abs(dx) > 24:
            continue
        contact = obj.get("estimated_contact_in_frames")
        if contact is None or contact <= 2:
            return True
    return False


def avoid_repeat(action, failures, state=None):
    """Replace a button press that already killed Mario within REPEAT_PX.

    A repeated fatal button is still replaced when an enemy is already touching
    Mario. Braking into that enemy is what the last level 1-2 deaths repeated,
    so a forward jump is tried instead of another step left.
    """
    if not failures:
        return action, None
    blocked = {
        failure["fatal_action"]
        for failure in failures
        if failure.get("fatal_action")
        and failure.get("px_before_death", APPROACH_PX + 1) <= REPEAT_PX
    }
    if action not in blocked:
        return action, None
    alternatives = ALTERNATIVES.get(action, ())
    if already_touching(state) and action in {"left", "left_jump"}:
        alternatives = ("right_run_jump", "jump", "right_run")
    for candidate in alternatives:
        if candidate not in blocked:
            return candidate, action
    return action, None


def correct_approach(action, state):
    """Cut short a jump that previously climbed high and fell into a hazard."""
    failures = state.get("prior_failures") or []
    mario = state.get("mario") or {}
    x, y = mario.get("x"), mario.get("y")
    if x is None or y is None:
        return action, None
    grounded = mario.get("grounded")
    held = state.get("jump_already_held")
    chosen = action
    reason = None
    commit_x = None
    for failure in failures:
        if failure.get("fall_height_px", 0) < HIGH_FALL_PX:
            continue
        start = int(failure["commit_x"])
        death = int(failure["death_x"])
        takeoff_y = int(failure["commit_y"])
        if not (start - 16 <= x <= death):
            continue
        stripped = WITHOUT_JUMP.get(chosen, chosen)
        if stripped == chosen:
            continue
        span = (
            LAUNCH_AFTER_COMMIT_PX
            if failure.get("fall_height_px", 0) >= SHORT_JUMP_PX
            else 16
        )
        if grounded and not held and start - 8 <= x <= start + span:
            chosen = stripped
            reason = "fatal_takeoff"
            commit_x = start
        elif (
            not grounded
            and failure.get("fall_height_px", 0) >= SHORT_JUMP_PX
            and takeoff_y - y >= HEIGHT_CAP_PX
            and x <= death
        ):
            chosen = stripped
            reason = "fatal_high_jump"
            commit_x = start
    if chosen == action:
        return action, None
    return chosen, {
        "rejected": action,
        "used": chosen,
        "reason": reason,
        "commit_x": commit_x,
    }


class FailureMemory:
    def __init__(self):
        self.lessons = []
        self._recent = deque(maxlen=RECENT_DECISIONS)

    @classmethod
    def load(cls, directory, env_id=None):
        memory = cls()
        directory = Path(directory)
        if not directory.exists():
            return memory
        collected = []
        for path in sorted(directory.glob("*.jsonl")):
            collected.extend(lessons_in_log(path, env_id))
        memory.lessons = collapse(collected)
        return memory

    def note(self, decision):
        if decision.get("action"):
            self._recent.append(decision)

    def finish_episode(self, summary):
        learned = None
        if (
            summary.get("stop_reason") == "terminated"
            and not summary.get("completed")
            and self._recent
            and self._recent[-1].get("action")
        ):
            fresh = lesson_from_episode(list(self._recent))
            self.lessons = collapse([*self.lessons, fresh])
            learned = next(
                lesson
                for lesson in self.lessons
                if lesson["fatal_action"] == fresh["fatal_action"]
                and lesson["death_x"] // 32 == fresh["death_x"] // 32
            )
        self._recent.clear()
        return learned

    def for_position(self, x):
        nearby = []
        x = int(x)
        for lesson in self.lessons:
            death = int(lesson["death_x"])
            commit = int(lesson.get("commit_x", death))
            start = min(death - APPROACH_PX, commit - COMMIT_LEAD_PX)
            if start <= x <= death:
                nearby.append(
                    {
                        **lesson,
                        "px_before_death": death - x,
                        "px_before_commit": commit - x,
                    }
                )
        nearby.sort(key=lambda lesson: lesson["px_before_death"])
        return nearby[:3]

    def attach(self, state):
        lessons = self.for_position(state["mario"]["x"])
        if lessons:
            state["prior_failures"] = lessons
        return state
