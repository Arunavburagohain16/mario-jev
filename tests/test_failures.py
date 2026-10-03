import json

from mario_jev.failures import (
    FailureMemory,
    avoid_repeat,
    correct_approach,
    hazard_from_state,
)


def death_decision(action="left_jump", x=1960, y=170):
    return {
        "type": "decision",
        "episode": 0,
        "action": action,
        "state": {
            "mario": {"x": x - 6, "y": 176, "motion": "grounded"},
            "nearby_objects": [
                {
                    "kind": "goomba",
                    "dx": 15,
                    "dy": 8,
                    "estimated_contact_in_frames": 0.0,
                    "defeated": False,
                }
            ],
        },
        "result": {"x": x, "y": y},
        "decisions": {
            "movement": "brake_left",
            "jump_pressed": True,
        },
    }


def test_hazard_prefers_the_nearest_enemy():
    hazard = hazard_from_state(death_decision()["state"])
    assert hazard == "goomba dx=15, dy=8, contact_frames=0.0"


def test_hazard_reports_a_pit_when_no_enemy_is_visible():
    hazard = hazard_from_state(
        {
            "mario": {"motion": "falling"},
            "nearby_objects": [],
            "terrain": {
                "summary": {"nearest_empty_column_below_feet": {"edge_distance_px": 16}}
            },
        }
    )
    assert hazard == "missing floor 16px ahead"


def test_log_load_keeps_deaths_and_ignores_other_endings(tmp_path):
    path = tmp_path / "run.jsonl"
    alive = death_decision("right_run", x=100)
    alive["result"]["terminated"] = False
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "config"}),
                json.dumps(death_decision()),
                json.dumps(
                    {
                        "type": "summary",
                        "episode": 0,
                        "stop_reason": "terminated",
                        "completed": False,
                    }
                ),
                json.dumps({**alive, "episode": 1, "decision": 3}),
                json.dumps(
                    {
                        "type": "summary",
                        "episode": 1,
                        "stop_reason": "decision_limit",
                        "completed": False,
                    }
                ),
                "{not json",
                "",
            ]
        )
        + "\n"
    )
    memory = FailureMemory.load(tmp_path)
    assert len(memory.lessons) == 1
    lesson = memory.lessons[0]
    assert lesson["death_x"] == 1960
    assert lesson["fatal_action"] == "left_jump"
    assert lesson["actions_before_death"] == ["left_jump"]
    assert lesson["times_seen"] == 1


def test_same_death_counts_once_per_matching_run(tmp_path):
    for index in range(2):
        path = tmp_path / f"run-{index}.jsonl"
        path.write_text(
            json.dumps(death_decision())
            + "\n"
            + json.dumps(
                {
                    "type": "summary",
                    "episode": 0,
                    "stop_reason": "terminated",
                    "completed": False,
                }
            )
            + "\n"
        )
    memory = FailureMemory.load(tmp_path)
    assert len(memory.lessons) == 1
    assert memory.lessons[0]["times_seen"] == 2


def test_lesson_appears_on_the_approach_and_not_before_it():
    memory = FailureMemory()
    memory.note(death_decision())
    learned = memory.finish_episode({"stop_reason": "terminated", "completed": False})
    assert learned["death_x"] == 1960
    early = {"mario": {"x": 1700}}
    memory.attach(early)
    assert "prior_failures" not in early
    close = {"mario": {"x": 1800}}
    memory.attach(close)
    assert close["prior_failures"][0]["px_before_death"] == 160
    assert close["prior_failures"][0]["fatal_action"] == "left_jump"


def test_a_finished_level_is_not_stored_as_a_death():
    memory = FailureMemory()
    memory.note(death_decision())
    assert (
        memory.finish_episode({"stop_reason": "completed", "completed": True}) is None
    )
    assert memory.lessons == []


def test_repeat_is_blocked_only_close_to_the_death():
    far = [{"fatal_action": "left_jump", "px_before_death": 160}]
    assert avoid_repeat("left_jump", far) == ("left_jump", None)
    near = [{"fatal_action": "left_jump", "px_before_death": 20}]
    assert avoid_repeat("left_jump", near) == ("left", "left_jump")
    both = [
        {"fatal_action": "left_jump", "px_before_death": 10},
        {"fatal_action": "left", "px_before_death": 10},
    ]
    assert avoid_repeat("left_jump", both) == ("right_run_jump", "left_jump")


def _step(x, y, grounded, action, held=False, enemy=False):
    objects = []
    if enemy:
        objects.append(
            {
                "kind": "goomba",
                "dx": 15,
                "dy": 8,
                "estimated_contact_in_frames": 0.0,
                "defeated": False,
            }
        )
    return {
        "action": action,
        "state": {
            "mario": {
                "x": x,
                "y": y,
                "grounded": grounded,
                "motion": "grounded" if grounded else "rising",
            },
            "jump_already_held": held,
            "nearby_objects": objects,
        },
        "result": {"x": x + 12, "y": max(26, y - 16)},
        "decisions": {"movement": "run_right", "jump_pressed": "jump" in action},
    }


def test_high_jump_before_death_is_the_commit_point():
    memory = FailureMemory()
    memory.note(_step(1749, 112, True, "right_run_jump"))
    memory.note(_step(1833, 26, False, "right_run_jump", held=True))
    memory.note(_step(1899, 112, True, "right_run"))
    memory.note(death_decision())
    learned = memory.finish_episode({"stop_reason": "terminated", "completed": False})
    assert learned["commit_x"] == 1749
    assert learned["commit_y"] == 112
    assert learned["fall_height_px"] >= 40
    early = {"mario": {"x": 1680}}
    memory.attach(early)
    assert early["prior_failures"][0]["commit_x"] == 1749
    state = {
        "mario": {"x": 1760, "y": 112, "grounded": True},
        "jump_already_held": False,
        "prior_failures": early["prior_failures"],
    }
    action, correction = correct_approach("right_run_jump", state)
    assert action == "right_run"
    assert correction["reason"] == "fatal_takeoff"
    airborne = {
        "mario": {"x": 1800, "y": 40, "grounded": False},
        "jump_already_held": True,
        "prior_failures": early["prior_failures"],
    }
    assert correct_approach("right_run_jump", airborne)[0] == "right_run"
    later = {
        "mario": {"x": 1904, "y": 176, "grounded": True},
        "jump_already_held": False,
        "prior_failures": early["prior_failures"],
    }
    assert correct_approach("right_run_jump", later)[0] == "right_run_jump"


def test_contact_still_replaces_the_fatal_button():
    failure = [{"fatal_action": "left_jump", "px_before_death": 8}]
    touching = death_decision()["state"]
    assert avoid_repeat("left_jump", failure, touching) == (
        "right_run_jump",
        "left_jump",
    )


def test_short_jump_is_changed_only_near_takeoff():
    memory = FailureMemory()
    memory.note(_step(135, 176, True, "right_run_jump"))
    memory.note(_step(160, 138, False, "right_run_jump", held=True))
    memory.note(_step(189, 176, True, "left_jump"))
    learned = memory.finish_episode({"stop_reason": "terminated", "completed": False})
    assert learned["commit_x"] == 135
    assert 32 <= learned["fall_height_px"] < 64
    takeoff = {
        "mario": {"x": 135, "y": 176, "grounded": True},
        "jump_already_held": False,
        "prior_failures": [learned],
    }
    assert correct_approach("right_run_jump", takeoff)[0] == "right_run"
    later = {
        "mario": {"x": 160, "y": 176, "grounded": True},
        "jump_already_held": False,
        "prior_failures": [learned],
    }
    assert correct_approach("right_run_jump", later)[0] == "right_run_jump"
