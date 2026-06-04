import streamlit as st
from openai import OpenAI
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Culture Content Generator",
    page_icon="🏆",
    layout="centered"
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stButton > button {
        background-color: #E8602C;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover { background-color: #C9511F; }
    .result-box {
        background: #FDEEE7;
        border-left: 4px solid #E8602C;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .safety-pass {
        background: #E1F5EE;
        border-left: 4px solid #0F6E56;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #085041;
    }
    .safety-flag {
        background: #FAEEDA;
        border-left: 4px solid #BA7517;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #633806;
    }
    .standard-badge {
        background: #1A3C5E;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .disclaimer {
        background: #F1EFE8;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-size: 0.78rem;
        color: #5F5E5A;
        margin-top: 1.5rem;
        line-height: 1.6;
    }
    .header-bar {
        background: linear-gradient(135deg, #1A3C5E, #E8602C);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── CTM Culture Data ──────────────────────────────────────────────────────────
ORG_NAME = "CTM Unlimited"
MISSION = "To transform organizations into equitable and sustainable environments where all people can thrive and contribute to company growth."
VISION = "To be the preferred trusted partner for companies looking to create and sustain legendary workplaces."
CULTURE_STATEMENT = "Empowering Excellence through Unity and Innovation."
TONE = "Bold, energizing, and human. Next level, legit, and deeply committed. Not corporate speak. Real talk."

CORE_VALUES = [
    "We are legit.",
    "We are next level in our delivery and listen first before we solve.",
    "We will always be different.",
    "We are committed to perpetual evolution.",
    "We level up by using tech and data.",
    "We challenge and disrupt false narratives.",
    "We are a team, the real kind, the good kind."
]

CULTURE_STANDARDS = {
    "Drive Results": "Consistently achieve high-quality outcomes by proactively solving problems and executing tasks efficiently. Seek solutions, deeply understand challenges, and prioritize timely execution.",
    "Be Prepared": "Maintain seamless continuity through regular interactions, clear communication, and meticulous documentation. Conduct regular touchpoints, send agendas in advance, compile follow-up notes.",
    "Leadership at Every Level": "Every team member is empowered to act as a leader, making informed decisions with confidence. Make informed decisions, know the business, and think strategically.",
    "Corporate Courage": "Demonstrate the courage to advise on ethical behaviors and best practices. Advise boldly, ensure compliance, and speak up when something is not right.",
    "Adaptability and Continuous Learning": "Embrace challenges and setbacks with resilience, fostering growth and continuous improvement. Show resilience, proactively adjust, and embrace a growth mindset.",
    "Next Level Innovation": "Drive competitive edge by bringing innovative ideas to life. Foster a culture of continuous improvement and encourage creative thinking.",
    "One Team": "Achieve collective goals through effective communication, problem-solving, and supporting each other. Share knowledge and support the team proactively.",
    "Ownership and Accountability": "Lead initiatives, see tasks through to completion, and take responsibility for outcomes. Take the lead, see things through, and admit mistakes as growth opportunities.",
    "Effective Communication": "Maintain alignment and ensure progress through clear, efficient, and continuous communication. Communicate clearly, actively participate, and ensure continuity.",
    "Legendary Workplace": "Anticipate needs, challenge false narratives, and proactively improve procedures and workflows. Embody a go-get-it mindset."
}

CONTENT_TYPES = [
    "Weekly all-staff culture message",
    "Manager talking points for team meeting",
    "Culture spotlight for newsletter",
    "New team member welcome message",
    "Town hall opening remarks",
    "Culture reminder for Slack or Teams",
    "Digital display board quote",
    "Friday reflection message"
]

values_text = "\n".join([f"- {v}" for v in CORE_VALUES])
standards_text = "\n".join([f"- {k}: {v}" for k, v in CULTURE_STANDARDS.items()])

# ── Load API key from Streamlit Secrets ───────────────────────────────────────
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except Exception:
    st.error("API key not found. Please add OPENAI_API_KEY to your Streamlit Secrets. Go to your app settings, click Secrets, and add: OPENAI_API_KEY = 'sk-your-key-here'")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "generated_content" not in st.session_state:
    st.session_state.generated_content = []
if "embeddings_store" not in st.session_state:
    st.session_state.embeddings_store = {}
if "embeddings_ready" not in st.session_state:
    st.session_state.embeddings_ready = False

# ── Helper: cosine similarity ─────────────────────────────────────────────────
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_best_standard(query_embedding, embeddings_store):
    best_standard, best_score = None, -1
    for standard, embedding in embeddings_store.items():
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            best_standard = standard
    return best_standard, round(best_score * 100, 1)

# ── Build embeddings on first load ────────────────────────────────────────────
if not st.session_state.embeddings_ready:
    with st.spinner("Loading culture database... this takes about 15 seconds on first visit."):
        try:
            embeddings_store = {}
            for standard, description in CULTURE_STANDARDS.items():
                doc = f"CTM Culture Standard: {standard}. {description}"
                response = client.embeddings.create(
                    input=doc,
                    model="text-embedding-3-small"
                )
                embeddings_store[standard] = response.data[0].embedding
            st.session_state.embeddings_store = embeddings_store
            st.session_state.embeddings_ready = True
        except Exception as e:
            st.error(f"Failed to build culture database: {str(e)}")
            st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <h2 style="margin:0; color:white; font-size:1.4rem;">🏆 Culture Content Generator</h2>
    <p style="margin:4px 0 0; opacity:0.85; font-size:0.9rem;">CTM Unlimited &nbsp;·&nbsp; Powered by VEAUX Professional Services</p>
</div>
""", unsafe_allow_html=True)

# ── Main app ──────────────────────────────────────────────────────────────────
embeddings_store = st.session_state.embeddings_store

st.markdown("### Generate culture content")

col1, col2 = st.columns(2)
with col1:
    content_type = st.selectbox("Content type", CONTENT_TYPES)
with col2:
    standard_option = st.selectbox(
        "Culture standard",
        ["Find best match automatically"] + list(CULTURE_STANDARDS.keys())
    )

additional_context = st.text_area(
    "Additional context (optional)",
    placeholder="E.g. 'The team just wrapped up a big project. Acknowledge the win and reinforce accountability.'",
    height=100
)

semantic_query = ""
if standard_option == "Find best match automatically":
    semantic_query = st.text_input(
        "Describe what you want the content to focus on",
        placeholder="E.g. 'taking responsibility' or 'working as one team' or 'being innovative'"
    )

if st.button("Generate content"):
    focus = semantic_query if standard_option == "Find best match automatically" else standard_option

    if not focus and standard_option == "Find best match automatically":
        st.warning("Please describe what you want or select a specific standard.")
    else:
        with st.spinner("Finding best match, generating content, and running safety review..."):

            # Semantic search
            if standard_option == "Find best match automatically":
                query_response = client.embeddings.create(
                    input=focus,
                    model="text-embedding-3-small"
                )
                query_embedding = query_response.data[0].embedding
                matched_standard, similarity = find_best_standard(query_embedding, embeddings_store)
                retrieved_context = CULTURE_STANDARDS[matched_standard]
            else:
                matched_standard = standard_option
                retrieved_context = CULTURE_STANDARDS[standard_option]
                similarity = 100.0

            # Generate content
            system_prompt = f"""You are a culture communications specialist for {ORG_NAME}.

Mission: {MISSION}
Vision: {VISION}
Culture Statement: {CULTURE_STATEMENT}

Core Values:
{values_text}

Most Relevant Culture Standard:
{matched_standard}: {retrieved_context}

All CTM Standards for reference:
{standards_text}

Tone: {TONE}

Rules:
- Ground content in CTM's actual culture standards
- Write bold, real, and next level, not corporate
- Never use dashes or em dashes
- No filler phrases
- Content must be inclusive and appropriate for all employees
- Write like a leader who believes every word"""

            user_message = f"""Create the following culture content for CTM:

Content Type: {content_type}
Primary Standard: {matched_standard}
Additional Context: {additional_context if additional_context else 'None provided.'}

Generate the content now."""

            gen_response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            generated = gen_response.choices[0].message.content

            # Safety review
            safety_response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=200,
                messages=[
                    {"role": "system", "content": "You are a responsible AI content safety reviewer for HR communications."},
                    {"role": "user", "content": f"""Review this workplace communication for bias, legal risk, exclusionary language, off-brand tone, and sensitive topics.

Content Type: {content_type}
Content: {generated}

Respond exactly:
SAFE: [YES or NO]
FLAGS: [comma-separated issues or NONE]
RECOMMENDATION: [one sentence]"""}
                ]
            )
            safety_text = safety_response.choices[0].message.content
            is_safe = "SAFE: YES" in safety_text.upper()
            flags = ""
            for line in safety_text.split("\n"):
                if line.startswith("FLAGS:") and "NONE" not in line.upper():
                    flags = line.replace("FLAGS:", "").strip()

            st.session_state.generated_content.insert(0, {
                "content_type": content_type,
                "standard": matched_standard,
                "similarity": similarity,
                "content": generated,
                "safe": is_safe,
                "flags": flags,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.generated_content:
    st.markdown("---")
    st.markdown("### Generated content")

    for item in st.session_state.generated_content:
        st.markdown(
            f'<span class="standard-badge">{item["standard"]}</span>'
            f'&nbsp;&nbsp;<small style="color:#888;">{item["content_type"]} &nbsp;·&nbsp; {item["timestamp"]}</small>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="result-box">{item["content"]}</div>', unsafe_allow_html=True)

        if item["safe"]:
            st.markdown('<div class="safety-pass">✅ Safety review passed. Requires human review before distribution.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safety-flag">⚠️ Review recommended. Flags: {item["flags"]}</div>', unsafe_allow_html=True)

        with st.expander("Copy raw text"):
            st.code(item["content"], language=None)

        st.markdown("---")

    # Download
    all_content = f"CTM CULTURE CONTENT\nGenerated by VEAUX Professional Services\n{'='*50}\n\n"
    for item in st.session_state.generated_content:
        all_content += f"CONTENT TYPE: {item['content_type']}\n"
        all_content += f"STANDARD: {item['standard']}\n"
        all_content += f"SAFETY: {'Approved' if item['safe'] else 'Review recommended'}\n"
        all_content += "-"*40 + "\n"
        all_content += item["content"] + "\n\n"
    all_content += "\n⚠️ All AI-generated content requires human review before distribution.\n"
    all_content += "Powered by CTM Culture Workbook + OpenAI GPT-4o + VEAUX Professional Services\n"

    st.download_button(
        label="Download all generated content",
        data=all_content,
        file_name=f"ctm_culture_content_{datetime.now().strftime('%Y-%m-%d')}.txt",
        mime="text/plain"
    )

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Human review required.</strong> All AI-generated content must be reviewed and approved before distribution to staff.<br>
🔒 <strong>Privacy.</strong> This app does not collect or store any personal data.<br><br>
Powered by CTM Culture Workbook &nbsp;·&nbsp; OpenAI GPT-4o &nbsp;·&nbsp; VEAUX Professional Services
</div>
""", unsafe_allow_html=True)
