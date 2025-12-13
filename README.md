# 🎙 AI-Based Speech to Personality Analysis 


## 🚀 Theme

AI for Impact – Open Innovation



>
> A privacy-preserving, multi-agent AI system that evaluates communication skills, confidence, and personality using Agentic AI & RAG.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Agentic](https://img.shields.io/badge/Architecture-Agentic%20AI-orange)](https://langchain-ai.github.io/langgraph/)
[![Privacy: Local](https://img.shields.io/badge/Privacy-100%25%20Local-green)](https://ollama.com/)

---

## 📌 Executive Summary
VocalPersona is an intelligent speech analysis platform built for the AI for Impact initiative. It helps students, job seekers, and professionals overcome communication barriers by providing objective, psychological-based feedback on their speech delivery.

Unlike standard tools that only analyze text, VocalPersona uses multi-modal agents to "hear" tone, pitch, and pauses—running entirely on local, open-source models to guarantee user privacy.


## ❗ Problem Statement

In today’s competitive academic and professional environment, effective communication skills are as important as technical knowledge. However, a large number of students, job seekers, and professionals struggle with expressing themselves clearly and confidently while speaking.

Some of the major challenges include:

- Lack of awareness about how their voice sounds to others  
- Difficulty identifying nervousness, hesitation, or low confidence in speech  
- Poor emotional expression and unclear tone during interviews or presentations  
- Fear of public speaking due to absence of constructive feedback  
- Limited access to personalized communication coaching  

Most existing solutions focus only on what is being said (content) and not how it is being said (delivery, tone, emotion, confidence). Human feedback is often subjective, inconsistent, or not easily accessible to everyone.

There is currently no simple, affordable, and scalable AI-based solution that can:
- Analyze speech objectively  
- Extract personality and communication traits  
- Provide instant, personalized improvement suggestions  

This communication gap directly impacts employability, confidence, and professional growth, especially for students and early-career professionals.


## 💡 The Solution: Agentic AI & RAG

VocalPersona solves this by orchestrating a team of specialized AI agents. We don't just ask one LLM to "analyze this." Instead, we treat speech analysis as a workflow:

1.  The Audio Engineer (Librosa): Extracts raw acoustic data (pitch variance, energy, pauses).
2.  The Psychologist (RAG Agent): Retrieves validated rules from a vector database (e.g., "Frequent pauses + low pitch variation = Perceived hesitation").
3.  The Coach (Guardrails): Ensures feedback is constructive and ethical, never medical.

---

## 🚀 Key Technical Innovations (Why We Win)

### 1. Multi-Agent Orchestration (LangGraph)
Instead of a single prompt, we use a *Cyclic Graph. Agents pass data to each other. If the *Communication Agent detects a long pause, it triggers the Emotion Agent to check if it was a "thoughtful pause" or a "nervous freeze."

### 2. RAG-Grounded Insights (Zero Hallucination)
We strictly prevent the AI from making up psychology.
* Bad AI: "You sound like a Leo." ❌
* VocalPersona: "Based on Psychology Today (2018), your rapid speech rate (160wpm) suggests high extraversion." ✅

### 3. Ethical Guardrails
We use Guardrails AI to filter outputs. The system strictly refuses to diagnose mental health conditions (e.g., anxiety, depression) and focuses solely on behavioral communication improvements.

---

## 🛠 Technology Stack

| Domain | Tech | Purpose |
| :--- | :--- | :--- |
| Orchestration | LangGraph, LangChain | Managing the multi-agent state workflow |
| LLMs (Local) | Ollama (Llama-3, Mistral) | Private, on-device reasoning |
| Audio Processing | OpenAI Whisper, Librosa | Transcribing text & extracting acoustic features |
| Vector DB | FAISS, Sentence-Transformers | Storing psychology rules for RAG |
| Backend | FastAPI, Pydantic | High-performance API |
| Quality Control | DeepEval, Guardrails AI | Testing for hallucinations and safety |

---

## 📂 Project Structure

```bash
├── app/
│   ├── main.py              # FastAPI Entry Point
│   └── api/                 # Endpoints (Upload/Analyze)
├── agents/
│   ├── workflow.py          # LangGraph Node Definitions
│   ├── communication.py     # Fluency Analysis Logic
│   └── emotion.py           # Acoustic Feature Extraction
├── rag/
│   ├── vector_store/        # FAISS Index (Psychology Rules)
│   └── ingestion.py         # PDF/Text Loader for Knowledge Base
├── guardrails/
│   ├── safety.xml           # Safety rules (No Medical Diagnosis)
│   └── validator.py         # Output sanitizer
└── tests/
    └── evals.py             # DeepEval Metric Tests





---

## 👥 Team Members

- 🧑‍💻 [*Praneeth*](https://github.com/gsmpraneeth)  
- 🧑‍💻 [*Yaswanth*](https://github.com/Yash14325)  
- 🧑‍💻 [*Mahesh*](https://github.com/kolli-mahesh)  
- 🧑‍💻 [*Dinesh*](https://github.com/dinesh9997)

---
