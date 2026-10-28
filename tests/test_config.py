"""The interviewer's setup screen can be built from config/*.yaml."""
from app.modes.interview import session as interview


def test_options_build():
    assert interview.options()["defaults"]
