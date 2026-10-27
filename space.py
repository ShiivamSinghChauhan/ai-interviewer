"""The Hugging Face Space entry point: Gradio SDK on ZeroGPU, the one free Space tier that still runs Python.

The app is plain FastAPI and never touches a GPU, but ZeroGPU only boots a Space that has registered a
@spaces.GPU function by the time a Gradio app launches. So this registers one that is never called, launches
an empty Gradio app just long enough to report it, and then serves the real app on the Space's port.

Hugging Face installs gradio and spaces for Gradio Spaces. Anywhere else, run `python -m app.main`.
"""
import os

import gradio as gr
import spaces

from scripts import download_models

# the public-demo settings from the Dockerfile; Space variables take precedence
DEFAULTS = {"HOST": "0.0.0.0", "PORT": "7860", "PUBLIC_DEMO": "1", "MAX_SESSION_MIN": "20",
            "STT_THREADS": "2", "TTS_THREADS": "2"}


@spaces.GPU
def unused() -> None:
    """Never called. ZeroGPU only checks that it exists."""


def report_to_zerogpu() -> None:
    """The spaces package sends ZeroGPU its startup report from inside the first Gradio launch."""
    with gr.Blocks() as placeholder:
        gr.Markdown("Starting...")
    placeholder.launch(server_port=7861, prevent_thread_lock=True, ssr_mode=False, quiet=True)  # 7860 is the app's
    placeholder.close()


if __name__ == "__main__":
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)
    report_to_zerogpu()
    # a Gradio Space has no build step to bake ~1 GB of models into, so each cold start downloads them
    download_models.main(["--kokoro", "fp32"])
    from app.main import main  # last: the settings read the environment above and look for the models

    main()
