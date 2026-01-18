import streamlit as st
import os
import tempfile
from modules.ingestion import DocumentProcessor
from modules.rag_engine import RAGEngine
from modules.voice import transcribe_audio
from streamlit_mic_recorder import mic_recorder

# Page Config
st.set_page_config(page_title="DocuBot: Voice & Text Assistant", layout="wide")

@st.cache_resource(show_spinner="Initializing...")
def get_rag_engine():
    return RAGEngine()

def main():
    st.title("DocuBot: Voice & Text Assistant")

    # Initialize session state
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = get_rag_engine()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: file upload & voice input
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"])
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing..."):
                try:
                    # Save uploaded file to temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Process file
                    docs = DocumentProcessor.load_document(tmp_path)
                    chunks = DocumentProcessor.split_documents(docs)
                    
                    # Update vector store
                    st.session_state.rag_engine.create_vector_store(chunks)
                    st.success("Document processed!")
                    
                    # Cleanup
                    os.remove(tmp_path)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.header("Voice Input")
        st.write("Click to record, click again to stop.")
        
        # Voice recorder widget
        audio_data = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=True,
            key='recorder'
        )

    # For input handling
    final_prompt = None

    # Check for voice input
    if audio_data:
        transcribed_text = transcribe_audio(audio_data['bytes'])
        if transcribed_text:
            final_prompt = transcribed_text
        else:
            st.warning("Could not understand audio.")

    # Check for text input (if no voice input processed)
    # Use chat_input regardless, but only process it if final_prompt is still None
    text_input = st.chat_input("Type your question here...")
    
    if text_input and not final_prompt:
        final_prompt = text_input

    # Display chat - render history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process new input
    if final_prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_engine.query(final_prompt)
                st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
