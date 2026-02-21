import streamlit as st
import tempfile
import json
import os
import sys
from pathlib import Path

os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
from workflow import create_patent_workflow
from config.models import PatentAnalysisState, DocumentInfo
from loguru import logger

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Patent Analysis App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .status-box {
        padding: 20px;
        border-radius: 12px;
        background: white;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .agent-active {
        border-left-color: #28a745;
        background-color: #f0fff4;
    }
    .agent-pending {
        border-left-color: #6c757d;
        color: #6c757d;
    }
    .metadata-card {
        background: #f1f3f5;
        padding: 10px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'workflow' not in st.session_state:
    st.session_state.workflow = create_patent_workflow()
if 'state' not in st.session_state:
    st.session_state.state = None
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Patent Shield")
    st.markdown("Advanced Technical Diagnostics")
    st.markdown("---")
    
    if st.button("🗑️ Reset Application"):
        st.session_state.state = None
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.subheader("🛠️ System Diagnostics")
    from agents import DescriptionReaderSubAgent, ClaimsReaderSubAgent, DrawingReaderSubAgent
    
    def check_agent(name, cls):
        try:
            available = getattr(sys.modules[cls.__module__], 'MINERU_AVAILABLE', False)
            return "✅ Initialized" if available else "❌ MinerU Missing"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    st.write("**SubAgents Status:**")
    st.write(f"- Description: {check_agent('Desc', DescriptionReaderSubAgent)}")
    st.write(f"- Claims: {check_agent('Claims', ClaimsReaderSubAgent)}")
    st.write(f"- Drawings: {check_agent('Draw', DrawingReaderSubAgent)}")
    
    if st.button("🔍 Run Engine Check"):
        st.write(f"Model Source: {os.environ.get('MINERU_MODEL_SOURCE', 'default')}")

# --- MAIN PAGE ---
st.title("🛡️ Patent Shield Analysis")

if not st.session_state.state:
    st.markdown("### 📝 1. Upload Patent Documents")
    st.info("Upload the components of your patent below. Only the **Description** is mandatory.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        desc_file = st.file_uploader("📑 Description (PDF/DOCX) *", type=["pdf", "docx"])
    with col2:
        claims_file = st.file_uploader("⚖️ Claims (PDF/DOCX)", type=["pdf", "docx"])
    with col3:
        drawings_file = st.file_uploader("🖼️ Drawings (PDF)", type=["pdf"])

    if desc_file:
        st.markdown("---")
        if st.button("🚀 Start Extraction & Analysis", type="primary", use_container_width=True):
            def save_tmp(uploaded):
                if not uploaded: return ""
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    return tmp.name

            # 1. Initialize State
            st.session_state.state = PatentAnalysisState(
                description_path=save_tmp(desc_file),
                claims_path=save_tmp(claims_file),
                drawings_path=save_tmp(drawings_file),
                description_info=DocumentInfo(filename=desc_file.name, file_size=desc_file.size, file_path=""),
                claims_info=DocumentInfo(filename=claims_file.name, file_size=claims_file.size, file_path="") if claims_file else None,
                drawings_info=DocumentInfo(filename=drawings_file.name, file_size=drawings_file.size, file_path="") if drawings_file else None
            )
            
            # 2. Automated Loop
            with st.status("🔍 Extracting technical data...", expanded=True) as status:
                try:
                    # Convert Pydantic model to dict for LangGraph input
                    initial_state = st.session_state.state.model_dump()
                    
                    # Invoke the compiled workflow check
                    # st.session_state.workflow is the COMPILED graph (app)
                    
                    # We can stream the events to show progress, or just invoke
                    # For simplicity let's just invoke the full graph
                    status.write("Running Extraction Pipeline...")
                    result = st.session_state.workflow.invoke(initial_state)
                    
                    # Update state with result
                    status.write("Pipeline Finished. Updating State...")
                    if isinstance(result, dict):
                        # Merge result back into Pydantic model
                        # Note: invoke returns the final state dict
                        # We should be careful not to overwrite the complex objects if they are missing
                        # But PatentAnalysisState should match the dict structure
                        
                        # Debug: Print keys returned
                        print(f"[App] Workflow returned keys: {list(result.keys())}")
                        
                        # Re-instantiate state from result dict
                        st.session_state.state = PatentAnalysisState(**result)
                    
                    status.update(label="✅ Extraction Complete!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="❌ Extraction Failed", state="error")
                    st.error(f"Pipeline Error: {str(e)}")
                    # Print full stack trace to terminal
                    import traceback
                    traceback.print_exc()

            st.rerun()

else:
    # --- RESULTS VIEW ---
    st.success("✅ Analysis Complete! Below is the consolidated state of the Document Reader Agent.")
    
    # Overview Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Description Status", "Extracted" if st.session_state.state.description_text else "Pending")
    m2.metric("Claims Status", "Extracted" if st.session_state.state.claims_text else "Skipped")
    m3.metric("Drawings Status", "Analyzed" if st.session_state.state.drawings_text else "Skipped")

    st.markdown("---")
    
    # Consolidated Content Tabs
    tab_d, tab_c, tab_dr = st.tabs(["📑 Description Text", "⚖️ Claims Text", "📊 Drawings Metadata & Text"])
    
    with tab_d:
        st.text_area("Final state: description_text", value=st.session_state.state.description_text, height=600)
        
    with tab_c:
        if st.session_state.state.claims_text:
            st.text_area("Final state: claims_text", value=st.session_state.state.claims_text, height=600)
        else:
            st.warning("No claims document was provided for extraction.")

    with tab_dr:
        if st.session_state.state.drawings_text:
            st.text_area("Final state: drawings_text", value=st.session_state.state.drawings_text, height=600)
        else:
            st.warning("No drawings document was provided for extraction.")

    # Metadata summary
    with st.expander("🛠️ View Raw State Metadata"):
        st.json(st.session_state.state.model_dump())

st.markdown("---")
st.caption("Patent Shield v1.2 | Multi-SubAgent Technical State View")

st.markdown("---")
st.caption("Patent Shield v1.1 | Powered by LangGraph & MinerU")
