import numpy as np

from app.config import load
from app.core import analysis
from app.core.turns import UserTurn

CFG = load("common")["analysis"]


def turn(text: str, voiced: float = 4.0, latency: float | None = 1.0, pauses=()) -> UserTurn:
    t = UserTurn(text=text, audio=np.zeros(1600, dtype=np.float32), voiced=voiced, latency=latency,
                 pauses=list(pauses))
    t.metrics = analysis.turn_metrics(t, CFG)
    return t


def test_restarts_and_fillers_are_removed_from_quotes():
    assert analysis.clean_disfluencies("I I w-w-want to, to go um there", ["um"]) == "I want to go there"


def test_turn_metrics():
    metrics = turn("so um the the model was uh slow", voiced=3.0, pauses=[0.4, 1.6]).metrics
    assert metrics["words"] == 5        # "so the model was slow": fillers and the repeat don't count
    assert metrics["wpm"] == 100        # 5 words in 3 s of speech
    assert metrics["fillers"] == 2
    assert metrics["long_pauses"] == 1  # only the 1.6 s one


def test_summary():
    turns = [turn("I built the retrieval layer", voiced=2.0, latency=0.8),
             turn("we measured recall at ten and it went up", voiced=3.0, latency=1.2, pauses=[2.0]),
             turn("yes", voiced=0.5, latency=None)]
    summary = analysis.summarize(turns)
    assert summary["answers"] == 3
    assert summary["response_time_s"] == 1.0   # median of the known latencies
    assert summary["words_per_answer"] == 5    # 15 words / 3 answers
    assert summary["long_pauses_per_min"] == 10.9  # 1 long pause in 5.5 s of speech
    assert [t["turn"] for t in summary["per_turn"]] == [1, 2, 3]


def test_no_metrics_no_summary():
    assert analysis.summarize([]) == {}
