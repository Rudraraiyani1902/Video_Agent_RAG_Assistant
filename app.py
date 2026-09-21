import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input, get_youtube_transcript
from core.transcriber import transcribe_all
from core.extractor import analyze_transcript
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Agent — Intelligent Meeting RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── High-Contrast Premium Design System ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Theme Reset & High-Contrast Colors ── */
:root {
    --bg-main: #070a13;
    --card-bg: rgba(15, 23, 42, 0.85);
    --border-color: rgba(255, 255, 255, 0.12);
    --border-glow: rgba(99, 102, 241, 0.45);
    --text-white: #ffffff;
    --text-bright: #f8fafc;
    --text-secondary: #e2e8f0;
    --text-muted: #94a3b8;
    --cyan-accent: #38bdf8;
    --indigo-accent: #818cf8;
    --purple-accent: #a855f7;
    --emerald-accent: #34d399;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-bright) !important;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #171f38 0%, var(--bg-main) 70%) !important;
    background-attachment: fixed !important;
}

/* ── Force ALL Markdown & Body Text to High-Contrast Legible Off-White ── */
.stMarkdown, 
.stMarkdown p, 
.stMarkdown li, 
.stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #f1f5f9 !important;
    font-size: 1.02rem !important;
    line-height: 1.8 !important;
}

[data-testid="stMarkdownContainer"] strong {
    color: #38bdf8 !important; /* Radiant light cyan for bold headers like 'Overview of RAG:' */
    font-weight: 700 !important;
}

[data-testid="stMarkdownContainer"] h1, 
[data-testid="stMarkdownContainer"] h2, 
[data-testid="stMarkdownContainer"] h3, 
[data-testid="stMarkdownContainer"] h4 {
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

/* ── Container Cards (st.container with border) ── */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 1.75rem !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.25s ease !important;
    margin-bottom: 1.25rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    border-color: var(--border-glow) !important;
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 28, 0.98) !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stSidebar"] label {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ── Form Inputs & Selectbox (Fixing White-on-White Issue) ── */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
.stTextInput input,
.stTextInput > div,
.stTextInput > div > div {
    background-color: #0f172a !important;
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.95rem !important;
}

.stTextInput input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background-color: transparent !important;
    padding: 0.65rem 0.85rem !important;
}

.stTextInput input::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

div[data-baseweb="input"]:focus-within,
.stTextInput input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.4) !important;
}

/* Selectbox */
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] * {
    background-color: #0f172a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
}

div[data-baseweb="select"] > div {
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
}

ul[data-baseweb="menu"],
ul[data-baseweb="menu"] li {
    background-color: #0f172a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Chat Input */
[data-testid="stChatInput"] textarea {
    background-color: #0f172a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.6rem;
    background: rgba(15, 23, 42, 0.7);
    padding: 0.45rem;
    border-radius: 14px;
    border: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2) !important;
}

.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.4rem !important;
    margin-bottom: 1rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.18) 0%, rgba(147, 51, 234, 0.18) 100%) !important;
    border: 1px solid rgba(129, 140, 248, 0.35) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
}

/* ── Status Bar / Pipeline Items ── */
.step-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0.9rem;
    background: rgba(15, 23, 42, 0.7);
    border-radius: 10px;
    border: 1px solid var(--border-color);
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}

.step-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.5; }
}

.spinner-ring {
    display: inline-block;
    width: 18px; height: 18px;
    border: 3px solid rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    border-top-color: #38bdf8;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ── Transcript Viewer ── */
.transcript-viewer {
    background: rgba(8, 12, 22, 0.9);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.85;
    color: #cbd5e1;
    max-height: 480px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Scrollbar */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Management ───────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "active_step": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.35rem 0.85rem;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:9999px;font-size:0.75rem;font-weight:600;color:#a5b4fc;text-transform:uppercase;margin-bottom:0.8rem;">⚡ Video Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.6rem;font-weight:800;letter-spacing:-0.02em;color:#ffffff;margin-bottom:0.2rem;">Video Agent</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#94a3b8;margin-bottom:1.5rem;">AI RAG Meeting Assistant</div>', unsafe_allow_html=True)
    
    source = st.text_input(
        "Source Media",
        placeholder="YouTube URL or local file path",
        help="Paste a YouTube link or local audio/video file path"
    )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("✨  Analyze Video", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.success("✅ Analysis Ready")
        with st.expander("Pipeline Workflow Stages", expanded=True):
            st.markdown("⚡ Instant Subtitles: **Done**")
            st.markdown("📝 Transcription: **Done**")
            st.markdown("🧠 Executive AI Brief: **Done**")
            st.markdown("🔍 RAG Vector Store: **Done**")

# ─── Main Content ───────────────────────────────────────────────────────────────
st.markdown('<div style="font-size:2.8rem;font-weight:800;letter-spacing:-0.03em;color:#ffffff;line-height:1.15;margin-bottom:0.4rem;">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:1.05rem;color:#94a3b8;margin-bottom:1.5rem;">Transform meetings, lectures, and YouTube videos into structured executive intelligence and interactive RAG chat.</div>', unsafe_allow_html=True)

# ─── Pipeline Execution ─────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("⚠️ Please enter a valid YouTube URL or local file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []

        try:
            with st.status("🚀 Processing media pipeline...", expanded=True) as status:
                st.write("⚡ **Step 1/4**: Checking for instant subtitles & acquiring media...")
                youtube_transcript = None
                if source.startswith("http://") or source.startswith("https://"):
                    youtube_transcript = get_youtube_transcript(source, language)

                st.write("📝 **Step 2/4**: Finalizing transcription...")
                if youtube_transcript:
                    transcript = youtube_transcript
                else:
                    chunks = process_input(source)
                    transcript = transcribe_all(chunks, language)

                st.write("🧠 **Step 3/4**: Generating executive brief, action items & decisions with Gemini...")
                analysis = analyze_transcript(transcript)

                st.write("🔍 **Step 4/4**: Indexing transcript into semantic vector store for Q&A...")
                rag_chain = build_rag_chain(transcript)

                status.update(label="✅ Video analysis complete! Explore your results below.", state="complete", expanded=False)

            st.session_state.result = {
                "title": analysis.get("title", "Meeting Analysis"),
                "transcript": transcript,
                "summary": analysis.get("summary", ""),
                "action_items": analysis.get("action_items", "No action items found."),
                "key_decisions": analysis.get("key_decisions", "No key decisions found."),
                "open_questions": analysis.get("open_questions", "No open questions found."),
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            time.sleep(0.5)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")

# ─── Results Dashboard ──────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    words_count = len(r['transcript'].split())

    # Executive Brief Header Card
    with st.container(border=True):
        col_t, col_m = st.columns([3, 1])
        with col_t:
            st.markdown('<div style="font-size:0.8rem;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.25rem;">📌 Session Title</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:1.8rem;font-weight:800;color:#ffffff;letter-spacing:-0.02em;">{r["title"]}</div>', unsafe_allow_html=True)
        with col_m:
            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex; gap:0.5rem; justify-content:flex-end; flex-wrap:wrap;">
                <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);padding:0.4rem 0.8rem;border-radius:8px;font-size:0.85rem;font-weight:600;color:#e2e8f0;">📝 {words_count:,} Words</span>
                <span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);padding:0.4rem 0.8rem;border-radius:8px;font-size:0.85rem;font-weight:600;color:#34d399;">⚡ RAG Ready</span>
            </div>
            """, unsafe_allow_html=True)

    # Clean Tabs Interface
    tab_summary, tab_actions, tab_chat, tab_transcript = st.tabs([
        "📋  Executive Summary",
        "⚡  Action Items & Decisions",
        "💬  Chat with Meeting",
        "📜  Full Transcript",
    ])

    # ── Tab 1: Executive Summary
    with tab_summary:
        with st.container(border=True):
            st.markdown("### 📋 Comprehensive Executive Summary")
            st.markdown(r['summary'])

    # ── Tab 2: Action Items & Decisions
    with tab_actions:
        col_act, col_dec = st.columns(2, gap="large")
        with col_act:
            with st.container(border=True):
                st.markdown("### ✅ Action Items & Deliverables")
                st.markdown(r['action_items'])

        with col_dec:
            with st.container(border=True):
                st.markdown("### 🔑 Key Decisions & Takeaways")
                st.markdown(r['key_decisions'])

        if r.get('open_questions') and "none" not in r['open_questions'].lower():
            with st.container(border=True):
                st.markdown("### ❓ Unresolved Follow-up Topics")
                st.markdown(r['open_questions'])

    # ── Tab 3: Interactive RAG Chat
    with tab_chat:
        with st.container(border=True):
            st.markdown("### 💬 Chat with your Meeting Transcript")
            st.markdown("<div style='font-size:0.95rem;color:#94a3b8;margin-bottom:1rem;'>Ask specific questions, request clarifications, or verify details anchored directly in the meeting transcript.</div>", unsafe_allow_html=True)

            chat_container = st.container()
            with chat_container:
                if not st.session_state.chat_history:
                    st.markdown("""
                    <div style="text-align:center; padding: 2.5rem 1rem; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1);">
                        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">💭</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">Ask anything about this video</div>
                        <div style="font-size: 0.9rem; color: #94a3b8; max-width: 420px; margin: 0.4rem auto 0;">
                            Type your question below or inquire about specific concepts, dates, and topics covered.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                            st.markdown(msg["content"])

            # Native chat input
            user_query = st.chat_input("Ask a question about this meeting or video...")

            if user_query:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with chat_container:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(user_query)

                with chat_container:
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Searching transcript context..."):
                            answer = ask_question(r["rag_chain"], user_query)
                            st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

            if st.session_state.chat_history:
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Chat History", type="secondary"):
                    st.session_state.chat_history = []
                    st.rerun()

    # ── Tab 4: Full Transcript
    with tab_transcript:
        with st.container(border=True):
            st.markdown("### 📜 Raw Transcript")
            st.markdown(f'<div class="transcript-viewer">{r["transcript"]}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Full Transcript (.txt)",
                data=r["transcript"],
                file_name=f"{r['title'].lower().replace(' ', '_')}_transcript.txt",
                mime="text/plain",
            )

else:
    # ─── Empty State ────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("""
        <div style="text-align: center; padding: 4rem 1rem;">
            <div style="display: inline-flex; justify-content: center; align-items: center; width: 70px; height: 70px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 20px; font-size: 2.2rem; margin-bottom: 1.25rem;">
                🎬
            </div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 0.6rem;">
                Ready to Analyze Your Meeting or Video
            </div>
            <div style="font-size: 1rem; color: #94a3b8; max-width: 500px; margin: 0 auto 2rem; line-height: 1.6;">
                Paste any YouTube URL or local media file in the sidebar and click <strong>Analyze Video</strong>.
            </div>
            <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap;">
                <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);padding:0.4rem 0.8rem;border-radius:8px;font-size:0.85rem;font-weight:600;color:#e2e8f0;">⚡ Instant Subtitles</span>
                <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);padding:0.4rem 0.8rem;border-radius:8px;font-size:0.85rem;font-weight:600;color:#e2e8f0;">📋 Executive Briefs</span>
                <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);padding:0.4rem 0.8rem;border-radius:8px;font-size:0.85rem;font-weight:600;color:#e2e8f0;">✅ Action Items</span>
                <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);padding:0.4rem 0.8rem;border-radius:8px;font-size:0.85rem;font-weight:600;color:#e2e8f0;">💬 RAG Context Chat</span>
            </div>
        </div>
        """, unsafe_allow_html=True)