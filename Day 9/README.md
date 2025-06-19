# 📄 AI Resume Optimizer (True Agentic Version)

This project is a sophisticated, multi-agent AI system built to solve a common problem for students and job seekers: tailoring a generic resume to a specific job description to increase the chances of getting an interview.

This version uses **LangGraph** to construct a true agentic workflow with a **decision-making loop**. This architecture directly addresses feedback to move beyond simple chains and implement agents in their "true form" by allowing the system to review its own work and self-correct.

## 🚀 Key Features

-   **True Agentic System**: Utilizes a team of specialized AI agents orchestrated by LangGraph, where each agent has a distinct role.
-   **Decision-Making Loop**: A "Reviewer Agent" assesses the quality of the initial analysis and can send the workflow back for a retry, demonstrating reasoning and self-correction.
-   **RAG-Powered JD Analysis**: The JD Alignment Agent uses Retrieval-Augmented Generation (RAG) to accurately extract key priorities from any job description.
-   **Intelligent Resume Rewriting**: The Resume Rewriter Agent uses the outputs from the other agents to craft a professionally tailored resume.
-   **Active Response Tracker**: The Response Tracker Agent analyzes feedback on successful resumes to provide actionable insights on "winning" phrases and sections.
-   **Transparent Workflow**: The Streamlit interface shows a real-time log of the agentic process, providing transparency into the system's operations.

## 🤖 The Agentic Team

1.  **Resume Content Analyzer Agent**: The first agent in the workflow. It reads and understands the candidate's base resume.
2.  **Analysis Reviewer Agent**: A quality control agent that reviews the work of the Analyzer Agent and decides if the workflow should continue or loop back for a retry.
3.  **JD Alignment Agent (RAG-Enabled)**: This agent ingests the target job description, creates a vector store in memory, and extracts the most critical skills and qualifications.
4.  **Resume Rewriter Agent**: The final content-generation agent. It synthesizes information from the previous agents to write the new, optimized resume.
5.  **Response Tracker Agent**: An active agent triggered by the user to analyze successful resumes and provide feedback on effective strategies.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **Agent Framework**: LangGraph
-   **LLM Orchestration**: LangChain
-   **LLM**: Google Gemini 1.5 Flash
-   **Vector Store (RAG)**: FAISS (in-memory)
-   **Embeddings**: Google `embedding-001`

## ⚙️ Setup and Installation

Follow these steps to get the project running locally.

### 1. Clone the Repository

```bash
git clone https://github.com/GowthamR7/Agentic-AI.git
cd Day 9
```
### 2. Create and activate environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run streamlit
```bash
streamlit run resume_optimizer_.py
```