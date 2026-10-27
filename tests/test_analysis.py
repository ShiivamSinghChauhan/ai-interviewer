import numpy as np

from app.config import load
from app.core import analysis
from app.core.turns import UserTurn

CFG = load("common")["analysis"]


def turn(text: str, voiced: float = 4.0, latency: float = 1.0, pauses=()) -> UserTurn:
    t = UserTurn(text=text, audio=np.zeros(1600, dtype=np.float32), voiced=voiced, latency=latency,
                 pauses=list(pauses))
    t.metrics = analysis.turn_metrics(t, CFG)
    return t


def test_disfluencies_are_removed_for_content_feedback():
    assert analysis.clean_disfluencies("I I w-w-want to, to go um there", ["um"]) == "I want to go there"


def test_disfluencies_are_counted():
    found = analysis.text_disfluencies("I I think w-w-we, um, wan-- wanted it", CFG)
    assert (found["whole_word"], found["part_word"], found["broken"], found["fillers"]) == (1, 1, 1, 1)


def test_syllables():
    assert analysis.syllables("make a simple guess") == 5


def test_long_pauses_count_as_possible_blocks():
    metrics = turn("so the the model was slow", pauses=[0.4, 1.6]).metrics
    assert metrics["blocks"] == [1.6]
    assert metrics["stutter_events"] == 2  # "the the" plus the block


def test_summary_and_trend():
    baseline = {"f0_median": None, "loudness_db": None, "rate_sps": 3.0}
    turns = [turn("I I I don't know", latency=4.0), turn("maybe um yes", latency=3.5),
             turn("I built the retrieval layer and tuned the chunk size", latency=0.8),
             turn("we measured recall at ten and it went up a lot", latency=0.6)]
    summary = analysis.summarize(turns, baseline, CFG)
    assert summary["turns"] == 4
    assert summary["stammer"]["repetitions_whole_word"] == 1
    trend = summary["trend"]
    assert trend["second_half"] < trend["first_half"]  # quicker, longer answers later on: they settled in
    assert 0 <= summary["nervousness"]["score"] <= 100


def test_no_metrics_no_summary():
    assert analysis.summarize([], {}, CFG) == {}
