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
    st.markdown("Advanced Patent Analysis Engine")
    st.markdown("---")
    
    st.subheader("📤 Upload Documents")
    desc_file = st.file_uploader("Description (PDF/DOCX/IMG) *", type=["pdf", "docx", "png", "jpg", "jpeg"])
    claims_file = st.file_uploader("Claims (PDF/DOCX/IMG)", type=["pdf", "docx", "png", "jpg", "jpeg"])
    drawings_file = st.file_uploader("Drawings (PDF/DOCX/IMG)", type=["pdf", "docx", "png", "jpg", "jpeg"])
    
    if desc_file:
        if st.button("🚀 Run Full Analysis", type="primary"):
            def save_tmp(uploaded):
                if not uploaded:
                    return ""
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    return tmp.name

            # 1. Initialize State
            st.session_state.state = PatentAnalysisState(
                description_path=save_tmp(desc_file),
                claims_path=save_tmp(claims_file),
                drawings_path=save_tmp(drawings_file),
                description_info=DocumentInfo(
                    filename=desc_file.name,
                    file_size=desc_file.size,
                    file_path=desc_file.name
                ),
                claims_info=DocumentInfo(
                    filename=claims_file.name,
                    file_size=claims_file.size,
                    file_path=claims_file.name
                ) if claims_file else None,
                drawings_info=DocumentInfo(
                    filename=drawings_file.name,
                    file_size=drawings_file.size,
                    file_path=drawings_file.name
                ) if drawings_file else None
            )
            st.session_state.history = []
            
            # 2. Automated Loop
            with st.status("🛠️ Running Patent Analysis Suite...", expanded=True) as status:
                agent_sequence = ["description_reader", "claims_reader", "drawing_reader"]
                for agent_name in agent_sequence:
                    status.update(label=f"🔄 Agent {agent_name.replace('_', ' ').title()} is working...")
                    try:
                        inputs = st.session_state.state.model_dump()
                        result = st.session_state.workflow.invoke(inputs)
                        
                        if isinstance(result, dict):
                            st.session_state.state = PatentAnalysisState(**result)
                        else:
                            st.session_state.state = result
                        
                        st.session_state.history.append(st.session_state.state.model_dump())
                        
                        # Stop if we hit the end (drawing_reader sets agent to "END" or similar)
                        if st.session_state.state.current_agent == "END":
                             break
                    except Exception as e:
                        st.error(f"❌ {agent_name} failed: {str(e)}")
                        st.session_state.state.errors.append(str(e))
                        break
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            
            st.rerun()
    else:
        st.info("💡 Description is mandatory to start analysis.")

    st.markdown("---")
    if st.session_state.state:
        st.write("**Analysis Status:** Complete" if st.session_state.state.processing_complete or st.session_state.state.current_agent == "END" else "**Analysis Status:** In Progress")
        
    if st.button("🗑️ Reset Application"):
        st.session_state.state = None
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.subheader("🛠️ System Diagnostics")
    from agents import DescriptionReaderSubAgent, ClaimsReaderSubAgent, DrawingReaderSubAgent
    
    def check_agent(name, cls):
        try:
            agent = cls()
            available = getattr(sys.modules[cls.__module__], 'MINERU_AVAILABLE', False)
            return "✅ Available" if available else "❌ MinerU Missing"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    st.write("**Agents Status:**")
    st.write(f"- Description: {check_agent('Desc', DescriptionReaderSubAgent)}")
    st.write(f"- Claims: {check_agent('Claims', ClaimsReaderSubAgent)}")
    st.write(f"- Drawings: {check_agent('Draw', DrawingReaderSubAgent)}")
    
    if st.button("🔍 Run Model Check"):
        cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_path.exists():
            st.write(f"HF Cache: Found at {cache_path}")
        else:
            st.write("HF Cache: Not found in home directory.")
        st.write(f"Model Source: {os.environ.get('MINERU_MODEL_SOURCE', 'default')}")

# --- MAIN PAGE ---
st.title("🛡️ Agentic Analysis Dashboard")

if not st.session_state.state:
    st.info("Please upload patent components in the sidebar to begin analysis.")
    
    # Showcase cards
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="status-box">
            <h3>📑 Description Reader</h3>
            <p>Extracts technical description using layout-aware OCR.</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class="status-box">
            <h3>⚖️ Claims Reader</h3>
            <p>Isolates legal claims from the uploaded claims document.</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class="status-box">
            <h3>🖼️ Drawing Reader</h3>
            <p>Processes drawings and associated text descriptions.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # --- WORKFLOW PROGRESS ---
    st.subheader("🏁 Execution Pipeline")
    
    # Define the agents in order
    agent_sequence = ["description_reader", "claims_reader", "drawing_reader"]
    
    progress_cols = st.columns(len(agent_sequence))
    
    for i, agent in enumerate(agent_sequence):
        with progress_cols[i]:
            is_active = st.session_state.state.current_agent == agent
            is_done = agent in [h.get("current_agent") for h in st.session_state.history] or \
                      (st.session_state.state.description_text != "" and agent == "description_reader") or \
                      (st.session_state.state.claims_text != "" and agent == "claims_reader") or \
                      (st.session_state.state.drawings_text != "" and agent == "drawing_reader")
            
            icon = "✅" if is_done else ("🔵" if is_active else "⚪")
            color = "#28a745" if is_done else ("#007bff" if is_active else "#6c757d")
            
            st.markdown(f"""
            <div style="text-align: center; border-bottom: 4px solid {color}; padding: 10px;">
                <span style="font-size: 1.5em;">{icon}</span><br>
                <b>{agent.replace('_', ' ').title()}</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- AGENT DATA VIEW ---
    col_content, col_meta = st.columns([2, 1])
    
    with col_content:
        st.subheader("📝 Extracted Content")
        tabs = st.tabs(["Description", "Claims", "Drawings"])
        
        with tabs[0]:
            if st.session_state.state.description_text:
                st.text_area("Extracted Description", value=st.session_state.state.description_text, height=500)
            else:
                st.info("Description pending extraction.")
                
        with tabs[1]:
            if st.session_state.state.claims_text:
                st.text_area("Extracted Claims", value=st.session_state.state.claims_text, height=500)
            else:
                st.info("Claims pending extraction.")

        with tabs[2]:
            if st.session_state.state.drawings_text:
                st.text_area("Extracted Drawings Content", value=st.session_state.state.drawings_text, height=500)
            else:
                st.info("Drawings pending extraction.")

    with col_meta:
        st.subheader("📊 Document Metadata")
        
        def show_doc_card(title, info):
            if info:
                st.markdown('<div class="status-box">', unsafe_allow_html=True)
                st.write(f"**📄 {title}**")
                st.markdown(f"""
                <div class="metadata-card">
                Filename: {info.filename}<br>
                Size: {info.file_size / 1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        show_doc_card("Description", st.session_state.state.description_info)
        show_doc_card("Claims", st.session_state.state.claims_info)
        show_doc_card("Drawings", st.session_state.state.drawings_info)
        
        # Errors
        if st.session_state.state.errors:
            st.error(" / ".join(st.session_state.state.errors))

st.markdown("---")
st.caption("Patent Shield v1.1 | Powered by LangGraph & MinerU")
