# CTM Culture Content Generator
### Built by Hope Malveaux — VEAUX Professional Services

A client-facing Streamlit web app that generates on-brand culture reinforcement content using OpenAI GPT-4o, semantic search (embeddings), RAG (ChromaDB), and built-in safety guardrails.

---

## How to Deploy (Free on Streamlit Cloud)

### Step 1 — Create a GitHub repository
1. Go to github.com and create a free account if you do not have one
2. Click **New repository**
3. Name it: `ctm-culture-generator`
4. Set to **Private** (recommended for client work)
5. Click **Create repository**

### Step 2 — Upload these files to GitHub
Upload all three files:
- `app.py` (rename `streamlit_app.py` to `app.py` before uploading)
- `requirements.txt`
- `README.md`

### Step 3 — Deploy on Streamlit Cloud
1. Go to **share.streamlit.io**
2. Sign in with your GitHub account
3. Click **New app**
4. Select your repository: `ctm-culture-generator`
5. Set main file path to: `app.py`
6. Click **Deploy**

Streamlit will build and deploy your app in 2 to 3 minutes.
You will get a public URL like: `https://ctm-culture-generator.streamlit.app`

---

## How to Share with a Client

Share the Streamlit URL with Melanie McBride or Vanessa Scott.
They open it in any browser, enter their own OpenAI API key, and start generating.

No installation. No Python. No Colab. Just a link.

---

## Adapting for a New Client

To adapt this for Houston Dynamo FC and Houston Dash (or any other client):

1. Open `app.py`
2. Update the culture profile section (ORG_NAME, MISSION, VISION, CULTURE_STANDARDS, etc.)
3. Push the update to GitHub
4. Streamlit auto-deploys the update within seconds

---

## Technology Stack

| Tool | Purpose |
|---|---|
| Streamlit | Web app framework |
| OpenAI GPT-4o | Content generation |
| text-embedding-3-small | Semantic search (find best culture standard) |
| ChromaDB | Vector store for RAG |
| Python | Backend logic, guardrails, safety review |

---

*Powered by the CTM Culture Workbook*
*Built by Hope Malveaux, VEAUX Professional Services*
*Postgraduate Certificate in Generative AI for Business Applications*
