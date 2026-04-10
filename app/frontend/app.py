"""
app/frontend/app.py

MedQA RAG Chatbot — Streamlit frontend.

Run with:
    streamlit run app/frontend/app.py

Requires backend running at localhost:8000:
    uvicorn app.main:app --reload

Features:
  - Chat interface with user/assistant bubbles
  - Concise / Detailed mode toggle
  - Top-K slider for retrieval tuning
  - PDF report upload (routes to /chat/report)
  - Tool badge on every response (RAG / Drug Tool / Report Tool)
  - Source chunks expandable panel
  - Source URL display for drug tool responses
  - Live backend health indicator
  - Session ID management with reset
"""

import uuid
import streamlit as st
from datetime import datetime

from api_client import check_health, send_message, send_report, ChatResult

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "MedQA Assistant",
    page_icon   = "🏥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS — dark medical/clinical theme ──────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.75rem 0 1.25rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.25rem;
}
.app-title {
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #e6edf3;
    font-family: 'IBM Plex Mono', monospace;
}
.app-subtitle {
    font-size: 0.75rem;
    color: #7d8590;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Health indicator ── */
.health-online  { color: #3fb950; font-size: 0.78rem; font-family: 'IBM Plex Mono', monospace; }
.health-offline { color: #f85149; font-size: 0.78rem; font-family: 'IBM Plex Mono', monospace; }

/* ── Chat messages ── */
.user-msg {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #e6edf3;
}
.bot-msg {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #388bfd;
    border-radius: 2px 12px 12px 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #cdd9e5;
}
.bot-msg-error {
    background: #1a1014;
    border: 1px solid #30363d;
    border-left: 3px solid #f85149;
    border-radius: 2px 12px 12px 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-size: 0.9rem;
    color: #f85149;
}

/* ── Tool badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-top: 6px;
    margin-right: 6px;
}
.badge-rag    { background: #1a2e4a; color: #388bfd; border: 1px solid #1f3c5e; }
.badge-drug   { background: #1e3a2f; color: #3fb950; border: 1px solid #2d5a3d; }
.badge-report { background: #2e2419; color: #d29922; border: 1px solid #4a3820; }

/* ── Timestamp ── */
.msg-time {
    font-size: 0.65rem;
    color: #484f58;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 4px;
}

/* ── Source URL ── */
.source-url {
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #7d8590;
    margin-top: 4px;
}
.source-url a { color: #388bfd; text-decoration: none; }
.source-url a:hover { text-decoration: underline; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Sidebar section labels ── */
.sidebar-label {
    font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #484f58;
    margin-bottom: 0.25rem;
    margin-top: 1rem;
}

/* ── Session ID box ── */
.session-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #7d8590;
    word-break: break-all;
    margin-bottom: 0.5rem;
}

/* ── PDF attachment preview ── */
.pdf-attached {
    background: #2e2419;
    border: 1px solid #4a3820;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #d29922;
    margin-bottom: 0.5rem;
}

/* ── Divider ── */
hr { border-color: #21262d; }

/* ── Scrollable chat container ── */
.chat-scroll {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 4px;
}

/* ── Input styling ── */
.stTextInput input, .stTextArea textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 2px rgba(56,139,253,0.15) !important;
}

/* ── Buttons ── */
.stButton button {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}
.stButton button:hover {
    background-color: #30363d !important;
    border-color: #388bfd !important;
}

/* ── Send button ── */
.stButton.send-btn button {
    background-color: #1f6feb !important;
    border-color: #1f6feb !important;
    color: white !important;
    font-weight: 500 !important;
}
.stButton.send-btn button:hover {
    background-color: #388bfd !important;
}

/* ── Radio / toggle ── */
.stRadio label { font-size: 0.85rem !important; color: #cdd9e5 !important; }
.stRadio div[role="radiogroup"] { gap: 0.5rem; }

/* ── Slider ── */
.stSlider { padding-top: 0.25rem; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #388bfd !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #7d8590 !important;
}

/* ── Source chunk cards ── */
.source-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.78rem;
    line-height: 1.5;
    color: #8b949e;
}
.source-card-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
}
.source-score { color: #3fb950; }
.source-name  { color: #7d8590; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────

def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]

if "session_id"   not in st.session_state: st.session_state.session_id   = _new_session_id()
if "messages"     not in st.session_state: st.session_state.messages     = []
if "mode"         not in st.session_state: st.session_state.mode         = "concise"
if "top_k"        not in st.session_state: st.session_state.top_k        = 5
if "pdf_bytes"    not in st.session_state: st.session_state.pdf_bytes    = None
if "pdf_filename" not in st.session_state: st.session_state.pdf_filename = None


# ── Tool badge helper ─────────────────────────────────────────────────────────

def _tool_badge(tool_used: str | None) -> str:
    if tool_used == "drug_tool":
        return '<span class="badge badge-drug">💊 Drug Tool</span>'
    elif tool_used == "report_tool":
        return '<span class="badge badge-report">📄 Report Tool</span>'
    else:
        return '<span class="badge badge-rag">🔬 RAG</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Health check
    st.markdown('<div class="sidebar-label">System</div>', unsafe_allow_html=True)
    if check_health():
        st.markdown('<span class="health-online">● Backend online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="health-offline">● Backend offline — start uvicorn</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Mode
    st.markdown('<div class="sidebar-label">Response Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        label     = "mode",
        options   = ["concise", "detailed"],
        index     = 0 if st.session_state.mode == "concise" else 1,
        format_func = lambda x: "⚡ Concise  (~8-12s)" if x == "concise" else "📋 Detailed  (~40-120s)",
        label_visibility = "collapsed",
    )
    st.session_state.mode = mode

    st.markdown("---")

    # Top-K
    st.markdown('<div class="sidebar-label">Retrieval — Top K Chunks</div>', unsafe_allow_html=True)
    top_k = st.slider(
        label     = "top_k",
        min_value = 1,
        max_value = 20,
        value     = st.session_state.top_k,
        label_visibility = "collapsed",
    )
    st.session_state.top_k = top_k
    st.caption(f"{top_k} chunks retrieved from Qdrant per query")

    st.markdown("---")

    # PDF Upload
    st.markdown('<div class="sidebar-label">📎 Upload Medical Report</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label       = "pdf_upload",
        type        = ["pdf"],
        label_visibility = "collapsed",
        help        = "Upload a PDF medical report. Will be analysed on next message send.",
    )
    if uploaded_file is not None:
        st.session_state.pdf_bytes    = uploaded_file.read()
        st.session_state.pdf_filename = uploaded_file.name
        st.markdown(
            f'<div class="pdf-attached">📄 {uploaded_file.name}<br>'
            f'<span style="color:#7d8590;font-size:0.65rem">'
            f'{round(len(st.session_state.pdf_bytes)/1024, 1)} KB — will send on next message</span></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.pdf_bytes is None:
        st.caption("No file attached")

    if st.session_state.pdf_bytes is not None:
        if st.button("✕ Clear PDF", use_container_width=True):
            st.session_state.pdf_bytes    = None
            st.session_state.pdf_filename = None
            st.rerun()

    st.markdown("---")

    # Session
    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="session-box">{st.session_state.session_id}</div>',
        unsafe_allow_html=True,
    )
    if st.button("↺ New Session", use_container_width=True):
        st.session_state.session_id   = _new_session_id()
        st.session_state.messages     = []
        st.session_state.pdf_bytes    = None
        st.session_state.pdf_filename = None
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#484f58;font-family:\'IBM Plex Mono\',monospace;">'
        'MedQA RAG Chatbot<br>Phase 3 — Local Dev<br>Qdrant · Llama 3.1 8B · MedQuAD</div>',
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🏥 MedQA Assistant</div>
        <div class="app-subtitle">RAG · Drug Lookup · Report Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat history
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:#484f58;">
            <div style="font-size:2rem;margin-bottom:0.75rem">🏥</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;margin-bottom:0.4rem;color:#7d8590">
                MedQA Assistant ready
            </div>
            <div style="font-size:0.75rem;color:#484f58">
                Ask a medical question · Mention a drug name · Upload a PDF report
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg">{msg["content"]}</div>'
                    f'<div class="msg-time" style="text-align:right;margin-right:4px">{msg["time"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Bot message
                if msg.get("error"):
                    st.markdown(
                        f'<div class="bot-msg-error">⚠ {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    badge = _tool_badge(msg.get("tool_used"))
                    source_url_html = ""
                    if msg.get("source_url"):
                        source_url_html = (
                            f'<div class="source-url">🔗 Source: '
                            f'<a href="{msg["source_url"]}" target="_blank">{msg["source_url"]}</a></div>'
                        )

                    st.markdown(
                        f'<div class="bot-msg">{msg["content"]}</div>'
                        f'<div style="margin-left:4px">{badge}{source_url_html}</div>'
                        f'<div class="msg-time">{msg["time"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Sources expander
                    sources = msg.get("sources", [])
                    if sources:
                        with st.expander(f"📚 {len(sources)} source chunks used"):
                            for i, chunk in enumerate(sources, 1):
                                score_pct = round(chunk.score * 100, 1)
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<div class="source-card-header">'
                                    f'<span class="source-name">#{i} · {chunk.source} · {chunk.category}</span>'
                                    f'<span class="source-score">score: {score_pct}%</span>'
                                    f'</div>'
                                    f'{chunk.chunk_text[:300]}{"..." if len(chunk.chunk_text) > 300 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )


# ── Input area ────────────────────────────────────────────────────────────────

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# Show PDF attached reminder above input
if st.session_state.pdf_bytes:
    st.markdown(
        f'<div class="pdf-attached" style="margin-bottom:0.4rem">'
        f'📄 PDF attached: <strong>{st.session_state.pdf_filename}</strong> '
        f'— will use Report Tool on send</div>',
        unsafe_allow_html=True,
    )

col_input, col_send = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        label       = "message",
        placeholder = "Ask a medical question, mention a drug, or upload a PDF report...",
        label_visibility = "collapsed",
        key         = "chat_input",
    )

with col_send:
    send_clicked = st.button("Send →", use_container_width=True, type="primary")

# Mode reminder caption
mode_label = "⚡ Concise" if st.session_state.mode == "concise" else "📋 Detailed"
st.caption(f"{mode_label} mode · Top-K: {st.session_state.top_k} · Session: `{st.session_state.session_id[:8]}...`")


# ── Send handler ──────────────────────────────────────────────────────────────

if send_clicked and user_input.strip():
    now = datetime.now().strftime("%H:%M:%S")

    # Add user message to history
    st.session_state.messages.append({
        "role"    : "user",
        "content" : user_input.strip(),
        "time"    : now,
    })

    # Call backend
    with st.spinner("Thinking..." if st.session_state.mode == "concise" else "Generating detailed response..."):
        if st.session_state.pdf_bytes:
            result: ChatResult = send_report(
                session_id = st.session_state.session_id,
                message    = user_input.strip(),
                pdf_bytes  = st.session_state.pdf_bytes,
                filename   = st.session_state.pdf_filename or "report.pdf",
                mode       = st.session_state.mode,
            )
            # Clear PDF after sending — one-shot
            st.session_state.pdf_bytes    = None
            st.session_state.pdf_filename = None
        else:
            result: ChatResult = send_message(
                session_id = st.session_state.session_id,
                message    = user_input.strip(),
                mode       = st.session_state.mode,
                top_k      = st.session_state.top_k,
            )

    now_reply = datetime.now().strftime("%H:%M:%S")

    if result.error:
        st.session_state.messages.append({
            "role"     : "assistant",
            "content"  : result.error,
            "time"     : now_reply,
            "error"    : True,
            "tool_used": None,
            "sources"  : [],
        })
    else:
        st.session_state.messages.append({
            "role"      : "assistant",
            "content"   : result.answer,
            "time"      : now_reply,
            "tool_used" : result.tool_used,
            "sources"   : result.sources,
            "source_url": result.source_url,
            "error"     : False,
        })

    st.rerun()