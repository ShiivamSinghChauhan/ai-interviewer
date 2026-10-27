"""Every mode's setup screen can be built from config/*.yaml, and the prompts fill without missing keys."""
from app.config import load
from app.modes import conversation, phone
from app.modes.interview import session as interview


def test_options_build_for_every_mode():
    for mode in (interview, conversation, phone):
        assert mode.options()["defaults"]


def test_conversation_labels_cover_every_choice():
    cfg = load("conversation")
    for table in ("modes", "levels", "lengths", "corrections"):
        assert set(cfg["labels"][table]) == set(cfg[table]), table


def test_surprise_person_is_valid():
    cfg = load("conversation")
    person = conversation.surprise()
    assert person["scene"] in cfg["scenes"] and person["mode"] in cfg["modes"]
    assert set(person["traits"]) == set(cfg["traits"])


def test_every_trait_has_five_levels():
    for trait, levels in load("conversation")["traits"].items():
        assert len(levels) == 5, trait


def test_phone_scenarios_are_complete():
    cfg = load("phone")
    for scenario in cfg["scenarios"]:
        assert scenario["category"] in cfg["categories"], scenario["id"]
        assert scenario["direction"] in ("incoming", "outgoing"), scenario["id"]
        if scenario["direction"] == "outgoing":
            assert scenario.get("user_task"), scenario["id"]
    picked = phone.pick_scenario(cfg, {"direction": "outgoing", "category": "any"})
    assert picked["direction"] == "outgoing"
