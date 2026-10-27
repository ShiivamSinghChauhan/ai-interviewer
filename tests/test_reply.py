from app.core.reply import SentenceChunker, parse_meta


def feed(text: str, step: int = 3) -> list[str]:
    """Stream `text` through a chunker a few characters at a time, like an LLM would."""
    chunker = SentenceChunker()
    out = []
    for i in range(0, len(text), step):
        out += chunker.push(text[i:i + step])
    return out + chunker.flush()


def test_meta_line_is_split_off():
    spoken, meta = parse_meta('Thanks for calling, bye!\n##META {"end": true}')
    assert spoken == "Thanks for calling, bye!"
    assert meta == {"end": True}


def test_missing_or_broken_meta():
    assert parse_meta("Just talking.") == ("Just talking.", {})
    assert parse_meta('Hmm. ##META {"end": tru') == ("Hmm.", {})


def test_first_piece_is_short_so_audio_starts_early():
    pieces = feed("Okay, got it. So when you scaled the retriever, what broke first, and how did you notice?")
    assert pieces[0] == "Okay, got it."
    assert " ".join(pieces).endswith("how did you notice?")


def test_nothing_after_the_meta_marker_is_spoken():
    pieces = feed('Right. That makes sense to me.\n##META {"advance": true}')
    assert pieces == ["Right.", "That makes sense to me."]


def test_short_sentences_wait_for_company():
    pieces = feed("Hmm, interesting. Yes. Tell me more about the evaluation set you built.")
    assert "Yes." not in pieces  # too short to be a clip of its own
    assert any("Yes." in piece for piece in pieces)
