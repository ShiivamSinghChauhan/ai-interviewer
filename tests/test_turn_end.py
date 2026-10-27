"""The end-of-turn vote, checked against the real patience presets in config/common.yaml."""
import pytest

from app.config import load
from app.core.turns import Signals, TurnConfig, end_reason


def preset(name: str) -> TurnConfig:
    values = load("common")["patience"][name]
    return TurnConfig(**{k: v for k, v in values.items() if k in TurnConfig.__dataclass_fields__})


STAMMER_FRIENDLY = preset("very_patient")


def test_a_block_mid_sentence_is_not_the_end():
    # sounded unfinished, the words say "not done", three seconds of silence
    signals = Signals(silence=3.0, smart_turn=0.2, semantic=0.1)
    assert end_reason(signals, STAMMER_FRIENDLY) is None


def test_a_trailing_and_never_counts_as_finished():
    # "...and" can sound final; it still has to wait for the longest pause
    signals = Signals(silence=STAMMER_FRIENDLY.answer_gap + 0.5, smart_turn=0.95, trailing=True)
    assert end_reason(signals, STAMMER_FRIENDLY) is None
    signals.silence = STAMMER_FRIENDLY.max_pause
    assert end_reason(signals, STAMMER_FRIENDLY) == "max_pause"


def test_a_confident_finish_waits_for_the_answer_gap():
    signals = Signals(silence=STAMMER_FRIENDLY.answer_gap - 0.1, smart_turn=0.9)
    assert end_reason(signals, STAMMER_FRIENDLY) is None
    signals.silence = STAMMER_FRIENDLY.answer_gap
    assert end_reason(signals, STAMMER_FRIENDLY) == "smart_turn"


def test_let_me_think_buys_extra_time():
    cfg = STAMMER_FRIENDLY
    assert end_reason(Signals(silence=cfg.max_pause + 1, smart_turn=0.1, thinking=True), cfg) is None
    assert end_reason(Signals(silence=cfg.thinking_pause, smart_turn=0.1, thinking=True), cfg) == "max_pause"


def test_the_words_can_finish_a_turn_the_audio_is_unsure_about():
    signals = Signals(silence=STAMMER_FRIENDLY.answer_gap, smart_turn=0.6, semantic=0.8)
    assert end_reason(signals, STAMMER_FRIENDLY) == "semantic"


def test_a_question_is_answered_quickly():
    signals = Signals(silence=0.8, smart_turn=0.6, question=True)
    assert end_reason(signals, STAMMER_FRIENDLY) == "question"


def test_only_a_long_sure_pause_overrides_the_words():
    almost = Signals(silence=STAMMER_FRIENDLY.answer_gap, smart_turn=0.9, semantic=0.1)
    assert end_reason(almost, STAMMER_FRIENDLY) is None
    almost.silence = STAMMER_FRIENDLY.answer_gap + 0.4
    assert end_reason(almost, STAMMER_FRIENDLY) == "smart_turn_override"


def test_a_slow_semantic_check_times_out():
    cfg = preset("patient")
    signals = Signals(silence=cfg.answer_gap, smart_turn=0.6, semantic_pending=True)
    assert end_reason(signals, cfg) == "semantic_timeout"


@pytest.mark.parametrize("name", ["very_patient", "patient", "normal"])
def test_every_preset_ends_a_turn_eventually(name):
    cfg = preset(name)
    assert end_reason(Signals(silence=cfg.max_pause, smart_turn=0.0), cfg) == "max_pause"


def test_quick_replies_skip_the_gap_in_normal_mode():
    signals = Signals(silence=0.3, smart_turn=0.9, quick=True)
    assert end_reason(signals, preset("normal")) == "smart_turn"
    assert preset("very_patient").quick_replies is False
