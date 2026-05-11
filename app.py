"""
Streamlit UI for SHL Assessment Recommender.
Run: streamlit run app.py
Requires the Flask backend running at API_URL.
"""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="centered",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.rec-card {
    background: #f8f9fa;
    border-left: 4px solid #0066cc;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.rec-card a { color: #0066cc; font-weight: 600; text-decoration: none; }
.rec-card a:hover { text-decoration: underline; }
.badge {
    display: inline-block;
    background: #0066cc;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    margin-left: 8px;
}
.type-legend {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎯 SHL Assessment Recommender")
st.caption("Describe the role you're hiring for and I'll recommend the right SHL assessments.")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "ended" not in st.session_state:
    st.session_state.ended = False

TYPE_LABELS = {
    "A": "Ability/Aptitude",
    "B": "Biodata",
    "C": "Competency",
    "K": "Knowledge/Skills",
    "P": "Personality",
    "S": "Simulation",
    "D": "Development",
    "E": "Exercise",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown("""
1. Describe the role you're hiring for
2. Answer any follow-up questions
3. Get a shortlist of SHL assessments
4. Ask to compare or refine anytime

**Example prompts:**
- *I'm hiring a mid-level Java developer*
- *Need assessments for a sales manager*
- *What's the difference between OPQ32r and MQ?*
""")
    st.divider()

    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            st.success("✅ Backend connected")
        else:
            st.error("❌ Backend error")
    except Exception:
        st.error("❌ Backend offline\nStart with:\n`python -m flask --app main run --port 8000`")

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.recommendations = []
        st.session_state.ended = False
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Recommendations panel ──────────────────────────────────────────────────────
if st.session_state.recommendations:
    st.divider()
    st.subheader(f"📋 Recommended Assessments ({len(st.session_state.recommendations)})")
    for rec in st.session_state.recommendations:
        type_label = TYPE_LABELS.get(rec.get("test_type", ""), rec.get("test_type", ""))
        st.markdown(f"""
<div class="rec-card">
    <a href="{rec['url']}" target="_blank">🔗 {rec['name']}</a>
    <span class="badge">{rec.get('test_type', '')}</span>
    <div class="type-legend">{type_label}</div>
</div>
""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
if st.session_state.ended:
    st.success("✅ Conversation complete. Click **New Conversation** to start over.")
else:
    user_input = st.chat_input("Describe the role or ask a question...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"messages": st.session_state.messages},
                        timeout=35,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    reply = data.get("reply", "Sorry, something went wrong.")
                    recs  = data.get("recommendations", [])
                    ended = data.get("end_of_conversation", False)

                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                    if recs:
                        st.session_state.recommendations = recs
                    st.session_state.ended = ended

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Make sure Flask is running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.rerun()
