"""
RAG System for Rijeesh Keloth's Publications
=============================================
A portfolio project demonstrating LLM integration with scientific publications.

This app allows users to ask questions about your research, and it will:
1. Search through your publications for relevant context
2. Use an LLM to generate accurate answers based on your actual papers

Deploy on Streamlit Cloud for free: https://streamlit.io/cloud
"""

import streamlit as st
import os
from typing import List, Dict
import json

# For embeddings and vector search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# For LLM API calls
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================
# PUBLICATION DATA
# ============================================================
# Your actual publications - add more as needed
PUBLICATIONS = [
    {
        "id": "solid_2024",
        "title": "Search for Very-Short-Baseline Oscillations of Reactor Antineutrinos with the SoLid Detector",
        "arxiv": "2407.14382",
        "year": 2024,
        "journal": "Physical Review Letters (submitted)",
        "collaboration": "SoLid",
        "abstract": """The SoLid collaboration conducted a search for sterile neutrinos using 280.3 live reactor-on days 
        of data from the BR2 reactor at SCK CEN in Belgium. 29,479 ± 603 antineutrino candidates were selected 
        with a signal-to-background ratio of 0.27. Machine learning tools were used for background suppression. 
        Model-dependent frequentist and Bayesian fits were performed to search for active-to-sterile oscillations. 
        No evidence of sterile neutrinos was found, providing constraints on the Reactor Antineutrino Anomaly.""",
        "your_contribution": """Co-led the sterile neutrino search analysis spanning 3 reactor cycles (2018-2021). 
        Developed the complete end-to-end pipeline from detector calibration through 47 systematic uncertainty 
        sources to publication submission. Responsible for data quality assurance and operations coordination.""",
        "keywords": ["sterile neutrino", "reactor", "antineutrino", "oscillation", "SoLid", "BR2", "machine learning", "Bayesian"]
    },
    {
        "id": "darkside_2024",
        "title": "DarkSide-20k sensitivity to light dark matter particles",
        "arxiv": "2407.xxxxx",
        "year": 2024,
        "journal": "Communications Physics 7, 422",
        "collaboration": "DarkSide-20k",
        "abstract": """DarkSide-20k is a next-generation dark matter detector using liquid argon technology. 
        This paper presents the sensitivity projections for detecting light dark matter particles. 
        The detector uses a dual-phase time projection chamber with 20 tonnes of low-radioactivity argon.""",
        "your_contribution": """Research Scientist responsible for wire-grid electrode development. 
        Built production control systems including 3 PyQt6 applications for precision assembly automation. 
        Developed spring-winding machine reducing cycle time by 75%. Integrated 48 load cells for real-time monitoring.""",
        "keywords": ["dark matter", "liquid argon", "TPC", "DarkSide-20k", "WIMP", "detector"]
    },
    {
        "id": "nova_2022",
        "title": "An Improved Measurement of Neutrino Oscillation Parameters by the NOvA Experiment",
        "arxiv": "2108.xxxxx",
        "year": 2022,
        "journal": "Physical Review D 106, 032004",
        "collaboration": "NOvA",
        "abstract": """NOvA presents improved measurements of neutrino oscillation parameters using 
        an exposure of 13.6×10^20 protons on target in neutrino mode and 12.5×10^20 in antineutrino mode. 
        The analysis uses convolutional neural networks for event classification and measures 
        theta23, delta_CP, and the mass hierarchy.""",
        "your_contribution": """Developed CNN-based particle identification algorithms for hadronic tau neutrino 
        interactions. Led complete short-baseline tau appearance analysis including 23 systematic uncertainty sources. 
        Served as Monte Carlo production expert, generating 45 million simulated neutrino interactions 
        (8×10^20 protons-on-target). Performed quality analysis on 344,000 APDs.""",
        "keywords": ["neutrino oscillation", "NOvA", "CNN", "tau neutrino", "Monte Carlo", "Fermilab"]
    },
    {
        "id": "solid_hnl_2024",
        "title": "Search for Heavy Neutral Leptons with the SoLid Detector",
        "arxiv": "2403.04662",
        "year": 2024,
        "journal": "In preparation",
        "collaboration": "SoLid",
        "abstract": """Analysis of SoLid data to search for Heavy Neutral Leptons (HNLs) in the MeV mass range. 
        Developed Boosted Decision Tree classifier achieving 100:1 background rejection while maintaining 
        80% signal efficiency. The analysis probes mixing between electron neutrinos and heavy sterile states.""",
        "your_contribution": """Developed BDT-based machine learning classifier for HNL signal-background discrimination. 
        Achieved background rejection factor of 100:1 while maintaining 80% signal efficiency. 
        Built complete analysis pipeline including systematic uncertainty quantification.""",
        "keywords": ["Heavy Neutral Lepton", "HNL", "sterile neutrino", "BDT", "machine learning", "SoLid"]
    },
    {
        "id": "solid_calibration",
        "title": "SoLid Detector Calibration and Performance",
        "arxiv": "2105.12984",
        "year": 2021,
        "journal": "JINST",
        "collaboration": "SoLid",
        "abstract": """The SoLid detector employs a novel hybrid scintillation technology using PVT scintillator 
        with LiF:ZnS(Ag) screens. The detector achieves channel-to-channel response controlled to a few percent, 
        energy resolution better than 14% at 1 MeV, and vertex determination with 5 cm precision.""",
        "your_contribution": """Led electromagnetic scintillation signal calibration for Phase II run period 
        (November 2020 - April 2021, 5 reactor cycles). Performed BiPo background studies analyzing 
        2.1×10^8 decay candidates, calibrating simulation models to 5% precision.""",
        "keywords": ["calibration", "scintillator", "energy resolution", "SoLid", "detector performance"]
    }
]

# Your professional information
PROFESSIONAL_INFO = """
Rijeesh Keloth is a Research Scientist at Virginia Tech working on the DarkSide-20k dark matter detection 
experiment. He has a PhD in Experimental Neutrino Physics from Cochin University of Science & Technology 
(India) and Fermilab (USA), with 6+ years of post-PhD experience.

Key achievements:
- 30+ peer-reviewed publications in Physical Review D, Physical Review Letters, JINST
- Co-leads wire-grid electrode development for DarkSide-20k (400+ scientists, 14 countries)
- Built production PyQt6/Python control systems reducing manufacturing cycle time by 75%
- Led data operations for SoLid experiment (60+ scientists, 3 European institutions)
- Developed ML classifiers (CNN, BDT) for particle physics applications
- Managed petabyte-scale data infrastructure with zero data loss

Technical expertise: Python, PyQt6, Machine Learning (BDT, CNN), Statistical Analysis, 
Monte Carlo Simulation, Data Acquisition Systems, Uncertainty Quantification
"""


# ============================================================
# RAG FUNCTIONS
# ============================================================

def search_publications(query: str, top_k: int = 3) -> List[Dict]:
    """
    Simple keyword-based search (works without embeddings).
    For production, use sentence-transformers for semantic search.
    """
    query_lower = query.lower()
    scores = []
    
    for pub in PUBLICATIONS:
        score = 0
        # Check title
        if any(word in pub["title"].lower() for word in query_lower.split()):
            score += 3
        # Check abstract
        abstract_words = pub["abstract"].lower()
        for word in query_lower.split():
            if word in abstract_words:
                score += 1
        # Check keywords
        for keyword in pub["keywords"]:
            if keyword.lower() in query_lower:
                score += 2
        # Check collaboration
        if pub["collaboration"].lower() in query_lower:
            score += 2
            
        scores.append((score, pub))
    
    # Sort by score and return top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    return [pub for score, pub in scores[:top_k] if score > 0]


def build_context(relevant_pubs: List[Dict]) -> str:
    """Build context string from relevant publications."""
    if not relevant_pubs:
        return PROFESSIONAL_INFO
    
    context_parts = [PROFESSIONAL_INFO, "\n\nRelevant Publications:\n"]
    
    for pub in relevant_pubs:
        context_parts.append(f"""
---
Title: {pub['title']}
Year: {pub['year']}
Journal: {pub['journal']}
Collaboration: {pub['collaboration']}
Abstract: {pub['abstract']}
Rijeesh's Contribution: {pub['your_contribution']}
---
""")
    
    return "".join(context_parts)


def query_llm(question: str, context: str, api_key: str, provider: str = "anthropic") -> str:
    """Query the LLM with context."""
    
    system_prompt = """You are an AI assistant helping users learn about Dr. Rijeesh Keloth's research 
and publications. Answer questions based on the provided context about his publications and contributions. 
Be specific and cite paper titles when relevant. If the context doesn't contain enough information 
to fully answer, say so honestly."""

    user_prompt = f"""Context about Rijeesh Keloth's research:
{context}

Question: {question}

Please provide a helpful, accurate answer based on the context above."""

    if provider == "anthropic" and ANTHROPIC_AVAILABLE:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    
    elif provider == "openai" and OPENAI_AVAILABLE:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content
    
    else:
        return "Error: No LLM API available. Please install anthropic or openai package."


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(
        page_title="Rijeesh Keloth - Research Scientist",
        page_icon="🔬",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .pub-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🔬 Rijeesh Keloth - Research Scientist</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about my research, publications, and expertise</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        provider = st.selectbox(
            "LLM Provider",
            ["anthropic", "openai"],
            help="Select which LLM API to use"
        )
        
        api_key = st.text_input(
            f"{provider.title()} API Key",
            type="password",
            help=f"Enter your {provider} API key"
        )
        
        st.divider()
        
        st.header("📚 Publications")
        st.write(f"**{len(PUBLICATIONS)}** papers indexed")
        
        for pub in PUBLICATIONS:
            with st.expander(f"{pub['collaboration']} ({pub['year']})"):
                st.write(f"**{pub['title']}**")
                st.write(f"*{pub['journal']}*")
                if pub['arxiv'] != "2407.xxxxx" and pub['arxiv'] != "2108.xxxxx":
                    st.write(f"[arXiv:{pub['arxiv']}](https://arxiv.org/abs/{pub['arxiv']})")
        
        st.divider()
        st.markdown("**Built with:**")
        st.markdown("- 🦜 LangChain / RAG")
        st.markdown("- 🤖 Claude / GPT")
        st.markdown("- 🎈 Streamlit")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Example questions
        example_questions = [
            "What is Rijeesh's contribution to the SoLid experiment?",
            "Tell me about the machine learning work in particle physics",
            "What is DarkSide-20k and what does Rijeesh do there?",
            "How many publications does Rijeesh have?",
            "Explain the sterile neutrino search",
        ]
        
        st.write("**Example questions:**")
        selected_example = st.selectbox("Select an example or type your own below:", 
                                        [""] + example_questions)
        
        question = st.text_area(
            "Your question:",
            value=selected_example,
            height=100,
            placeholder="Ask anything about my research, publications, or expertise..."
        )
        
        if st.button("🔍 Get Answer", type="primary"):
            if not question:
                st.warning("Please enter a question.")
            elif not api_key:
                st.warning(f"Please enter your {provider} API key in the sidebar.")
            else:
                with st.spinner("Searching publications and generating answer..."):
                    # Step 1: Search relevant publications
                    relevant_pubs = search_publications(question)
                    
                    # Step 2: Build context
                    context = build_context(relevant_pubs)
                    
                    # Step 3: Query LLM
                    try:
                        answer = query_llm(question, context, api_key, provider)
                        
                        st.success("Answer generated!")
                        st.markdown("### 📝 Answer")
                        st.markdown(answer)
                        
                        # Show sources
                        if relevant_pubs:
                            st.markdown("### 📚 Sources Used")
                            for pub in relevant_pubs:
                                st.markdown(f"""
                                <div class="pub-card">
                                <strong>{pub['title']}</strong><br>
                                <em>{pub['journal']} ({pub['year']})</em><br>
                                Collaboration: {pub['collaboration']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.header("👤 About")
        st.markdown("""
        **Rijeesh Keloth, Ph.D.**
        
        Research Scientist @ Virginia Tech
        
        🔬 **Current:** DarkSide-20k dark matter detection
        
        📊 **Expertise:**
        - Machine Learning (CNN, BDT)
        - Statistical Analysis
        - Production Software (PyQt6)
        - Large-scale Data Systems
        
        📄 **Publications:** 30+
        
        🌍 **Collaborations:**
        - DarkSide-20k (400+ scientists)
        - SoLid (60+ scientists)
        - NOvA (Fermilab)
        """)
        
        st.divider()
        
        st.markdown("""
        **📫 Contact**
        - Email: rijeesh@vt.edu
        - [LinkedIn](https://linkedin.com/in/rijeeshkeloth)
        - [INSPIRE-HEP](https://inspirehep.net/authors/1454963)
        """)


if __name__ == "__main__":
    main()
