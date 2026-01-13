# 🔬 Rijeesh Keloth - Research Assistant (RAG)

A **Retrieval-Augmented Generation** application that allows anyone to ask questions about my research publications and get accurate, context-aware answers.

## 🎯 What This Demonstrates

This project showcases:
- **LLM API Integration** (Claude/OpenAI) in production
- **RAG Architecture** - retrieval + generation pipeline
- **Prompt Engineering** for accurate responses
- **Web Application Development** with Streamlit

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/rijeeshkeloth/research-rag.git
cd research-rag
pip install -r requirements.txt
```

### 2. Get an API Key

Choose one:
- **Anthropic Claude**: https://console.anthropic.com/
- **OpenAI GPT**: https://platform.openai.com/

### 3. Run Locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app" → Select your repo
4. Deploy!

Users will enter their own API keys in the sidebar.

## 📁 Project Structure

```
research-rag/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── publications/      # (Optional) PDF storage
    ├── solid_2024.pdf
    └── ...
```

## 🔧 How It Works

```
User Question
     │
     ▼
┌─────────────────┐
│  1. RETRIEVAL   │  Search publications for relevant context
│  (Keyword/Embed)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. AUGMENT     │  Build prompt with retrieved context
│  (Context Build)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. GENERATE    │  LLM generates answer from context
│  (Claude/GPT)   │
└────────┬────────┘
         │
         ▼
    Answer + Sources
```

## 📚 Publications Indexed

| Paper | Year | Collaboration | My Role |
|-------|------|---------------|---------|
| Search for Very-Short-Baseline Oscillations... | 2024 | SoLid | Analysis Lead |
| DarkSide-20k sensitivity to light dark matter | 2024 | DarkSide-20k | Software Lead |
| Improved Measurement of Neutrino Oscillations | 2022 | NOvA | ML Developer |
| Search for Heavy Neutral Leptons | 2024 | SoLid | ML Classifier |
| SoLid Detector Calibration | 2021 | SoLid | Calibration Lead |

## 🛠️ Customization

### Add More Publications

Edit the `PUBLICATIONS` list in `app.py`:

```python
PUBLICATIONS.append({
    "id": "new_paper",
    "title": "Your Paper Title",
    "arxiv": "2401.xxxxx",
    "year": 2024,
    "journal": "Journal Name",
    "collaboration": "Collaboration",
    "abstract": "Paper abstract...",
    "your_contribution": "What you did...",
    "keywords": ["keyword1", "keyword2"]
})
```

### Use Semantic Search (Recommended)

For better search quality, uncomment the embedding code and install:

```bash
pip install sentence-transformers chromadb
```

### Add PDF Processing

To ingest actual PDFs:

```bash
pip install pypdf langchain
```

Then use LangChain's PDF loader to chunk and embed documents.

## 🎨 Screenshots

*Add screenshots of your deployed app here*

## 📝 Example Questions

- "What is your contribution to the SoLid experiment?"
- "Tell me about the machine learning work you've done"
- "What is DarkSide-20k?"
- "Explain the sterile neutrino search"
- "What systematic uncertainties did you handle?"

## 🔒 Security Notes

- API keys are entered by users, not stored
- No sensitive data in the codebase
- Streamlit Cloud handles secrets securely

## 📫 Contact

- **Email**: rijeesh@vt.edu
- **LinkedIn**: [linkedin.com/in/rijeeshkeloth](https://linkedin.com/in/rijeeshkeloth)
- **INSPIRE-HEP**: [inspirehep.net/authors/1454963](https://inspirehep.net/authors/1454963)

---

*Built as a portfolio project demonstrating LLM integration and RAG architecture.*
