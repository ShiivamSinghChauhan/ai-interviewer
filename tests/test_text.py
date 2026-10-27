"""Text handling: delivery cues, LLM chain entries and JSON replies, the jargon glossary."""
import pytest

from app.audio import cues
from app.llm.providers import extract_json, parse_entry
from app.modes.interview.glossary import fix_transcript


def test_cues_are_stripped_for_plain_voices_and_captions():
    assert cues.plain("[sigh] Well, <emphasis>that</emphasis> broke [break] again.") == "Well, that broke again."
    assert cues.plain("[chuckling]") == ""


def test_angle_tags_become_bracket_cues_and_are_capped():
    assert cues.normalise("It was <stress>really</stress> slow.") == "It was [emphasis] really slow."
    assert cues.normalise("[a] one [b] two [c] three [d] four", max_cues=2) == "[a] one [b] two three four"


def test_respelling_matches_whole_words_only():
    mapping = {"LLM": "L L M", "Qdrant": "quadrant"}
    assert cues.respell("LLMs and vLLM on Qdrant", mapping) == "LLMs and vLLM on quadrant"


def test_only_some_engines_perform_cues():
    assert cues.supports("fish")
    assert cues.supports("elevenlabs", "eleven_v4_turbo")
    assert not cues.supports("elevenlabs", "eleven_multilingual_v2")
    assert not cues.supports("edge")


def test_chain_entries_carry_typed_params():
    assert parse_entry("groq:openai/gpt-oss-120b|reasoning_effort=low|top_p=0.9|n=2|stream=false") == (
        "groq", "openai/gpt-oss-120b", {"reasoning_effort": "low", "top_p": 0.9, "n": 2, "stream": False})
    assert parse_entry("openrouter:google/gemma-4-31b-it:free")[1] == "google/gemma-4-31b-it:free"


def test_json_is_found_inside_chatter_and_fences():
    assert extract_json('Sure! ```json\n{"questions": [{"q": "Why?"}]}\n``` Hope that helps.') == {
        "questions": [{"q": "Why?"}]}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        extract_json("I could not do that.")


@pytest.mark.parametrize("heard, meant", [
    ("we used q laura on a single gpu", "we used QLoRA on a single GPU"),
    ("fine tuned it with laura", "fine-tuned it with LoRA"),
    ("served with v l l m and paged detention", "served with vLLM and paged attention"),
    ("scored by an llm as a judge", "scored by an LLM-as-a-judge"),
    ("a rg pipeline with bm twenty five and a re ranker", "a RAG pipeline with BM25 and a reranker"),
    ("stored in q drant", "stored in Qdrant"),
    ("quantized to int four", "quantized to INT4"),
])
def test_jargon_is_unmangled(heard, meant):
    assert fix_transcript(heard) == meant


def test_rope_is_only_rope_when_talking_about_positions():
    assert fix_transcript("rope scaling stretches the context length") == "RoPE scaling stretches the context length"
    assert fix_transcript("we tied a rope") == "we tied a rope"
