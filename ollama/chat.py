import streamlit as st
import requests
import json

# Function to simulate Ollama API call
def get_ollama_response_stream(prompt, model_name="llama2"):
    """
    Simulates a call to an Ollama server, yielding chunks for streaming.
    Replace this with your actual Ollama API endpoint and logic for streaming.
    """
    try:
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        # Set stream to True for actual streaming from Ollama
        data = {"model": model_name, "prompt": prompt, "stream": True}
        
        with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        json_data = json.loads(line.decode('utf-8'))
                        # Ollama sends chunks with 'response' field
                        chunk = json_data.get('response', '')
                        yield chunk
                        if json_data.get('done'):
                            break
                    except json.JSONDecodeError:
                        # Handle cases where a line might not be a complete JSON object
                        continue
    except requests.exceptions.ConnectionError:
        yield "Error: Could not connect to Ollama. Make sure it's running."
    except requests.exceptions.RequestException as e:
        yield f"Error during Ollama request: {e}"

# Function to get available Ollama models (simulated)
def get_ollama_models_list():
    """
    Simulates fetching a list of available Ollama models.
    In a real application, this would call Ollama's /api/tags endpoint.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models_data = response.json()
        # Extract model names from the response (e.g., {"models": [{"name": "llama2"}, {"name": "mistral"}]})
        return [m['name'].split(':')[0] for m in models_data.get('models', []) if 'name' in m]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama to fetch models. Is Ollama running?")
        return ["llama2"] # Default fallback model
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Ollama models: {e}")
        return ["llama2"] # Default fallback model

# Set basic page configuration
st.set_page_config(
    page_title="Ollama Chat App",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Use st.header for a more prominent title
st.header("🤖 Ollama Chat Assistant")

# Add a subtle divider for visual separation
st.divider()

# Model Selection Dropdown moved to sidebar
with st.sidebar:
    st.header("Model Settings") # Title for the sidebar section
    available_models = get_ollama_models_list()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = available_models[0] if available_models else "llama2"

    st.session_state.selected_model = st.selectbox(
        "Choose an Ollama Model:",
        options=available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        key="model_selector"
    )
    st.info(f"Currently using: **{st.session_state.selected_model}**")
    st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello there! How can I help you today? ⚡"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask Ollama a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Use st.write_stream to handle the generator output
        # It automatically handles the "typewriter" effect and cursor
        full_response = st.write_stream(get_ollama_response_stream(prompt, st.session_state.selected_model))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})