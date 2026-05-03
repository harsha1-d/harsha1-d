# Ollama Beginners Series

A minimal Python project that shows how to call a local Ollama model.

## Prerequisites

- Python 3.11 or newer
- A running Ollama server on `http://127.0.0.1:11434`
- The `requests` package installed

## Install dependencies

```bash
python -m pip install requests streamlit
```

## Run the examples

### Basic example (simple one-shot generation):

```bash
python basic.py
```

### Chat app (interactive Streamlit interface):

```bash
streamlit run chat.py
```

Then open your browser to `http://localhost:8501`.

## Files in this project

- `basic.py` — simple one-shot prompt generation
- `chat.py` — interactive Streamlit chat interface with model selection and streaming
- `pyproject.toml` — project metadata and dependencies
- `.python-version` — recommended Python version for pyenv
- `.webui_secret_key` — placeholder secret key file
- `uv.lock` — placeholder lock file for the project

## Notes

- If your local Ollama server uses a different model name, update `DEFAULT_MODEL` in `basic.py` or select a different model in the chat.py sidebar.
- Make sure Ollama is running before launching the app.
- For the chat app, it supports streaming responses and model selection from your local Ollama instance.