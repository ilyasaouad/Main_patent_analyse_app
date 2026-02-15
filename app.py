import streamlit as st
import tempfile
from pathlib import Path
from agents.document_reader_agent.agent import DocumentReaderAgent

st.set_page_config(page_title="Patent Analysis App", layout="wide")

st.title("🛡️ Patent Analysis Orchestrator")
st.markdown("---")

# Initialize our Agent
if 'agent' not in st.session_state:
    st.session_state.agent = DocumentReaderAgent()

uploaded_file = st.file_uploader("Drop a Patent (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    # 1. State Management: The 'input' for our graph
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        doc_path = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Step 1: Document Reader Agent")
        if st.button("Invoke Extraction Agent", type="primary"):
            with st.spinner("Agent is extracting content..."):
                text, file_path = st.session_state.agent.run(doc_path)
                st.session_state.extracted_text = text
                st.session_state.result_path = file_path
                st.success("Extraction Completed!")

    with col2:
        st.subheader("🧠 Step 2: Analysis Agent (Future)")
        st.info("This agent will receive the output from Stage 1.")
        if st.button("Invoke Analysis Agent", disabled='extracted_text' not in st.session_state):
            st.write("Processing analysis logic...")

    # Results Display
    if 'extracted_text' in st.session_state:
        st.markdown("---")
        st.write(f"📂 **Agent Output File:** `{st.session_state.result_path}`")
        with st.expander("View Agent Output"):
            st.text_area("Extracted Markdown:", value=st.session_state.extracted_text, height=400)
