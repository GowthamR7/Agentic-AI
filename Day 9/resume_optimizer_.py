import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import uuid
import time

# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔴 GOOGLE_API_KEY not found. Please create a .env file and add your key.")
        st.stop()
    genai_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.4)
except Exception as e:
    st.error(f"🔴 Failed to configure Google AI. Error: {e}")
    st.stop()


# --- 2. LANGGRAPH STATE DEFINITION (AGENT'S SHARED MEMORY) ---

class ResumeOptimizerState(TypedDict):
    """Defines the structure of the shared memory for the agents."""
    base_resume_text: str
    job_description: str
    resume_analysis: str
    jd_priorities: str
    optimized_resume: str
    workflow_log: List[str]
    retry_count: int
    review_decision: str


# --- 3. AGENT NODE DEFINITIONS ---

def resume_content_analyzer_agent(state: ResumeOptimizerState):
    """Agent 1: Analyzes the base resume. Can be told to retry."""
    state['workflow_log'].append("▶️ Agent 1: Resume Content Analyzer - Analyzing base resume...")
    state['retry_count'] += 1
    
    prompt = ChatPromptTemplate.from_template(
        """You are a professional resume analyzer. Based on the following resume text, provide a concise summary of the candidate's key skills, tools, project highlights, and overall strengths. Make the summary clear and well-structured.

        Resume Text:
        {resume}
        
        Analysis Summary:"""
    )
    chain = prompt | genai_llm
    analysis = chain.invoke({"resume": state["base_resume_text"]})
    state["resume_analysis"] = analysis.content
    return state

def analysis_reviewer_agent(state: ResumeOptimizerState):
    """
    New Agent: Reviews the work of the first agent. This demonstrates reasoning.
    """
    state['workflow_log'].append("▶️ New Agent: Analysis Reviewer - Checking quality of resume analysis...")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a quality control reviewer. Review the following resume analysis. Is it detailed, well-structured, and useful? 
        Answer with only the single word 'YES' or 'NO'.

        Resume Analysis to Review:
        {resume_analysis}

        Decision (YES or NO):"""
    )
    chain = prompt | genai_llm
    decision = chain.invoke({"resume_analysis": state["resume_analysis"]})
    state["review_decision"] = decision.content.strip().upper()
    return state

def jd_alignment_agent(state: ResumeOptimizerState):
    """Agent 2: Extracts priorities from the JD using RAG."""
    state['workflow_log'].append("▶️ Agent 2: JD Alignment (RAG) - Extracting priorities from Job Description...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(state["job_description"])
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()
    except Exception as e:
        st.error(f"🔴 Failed to create RAG vector store. Error: {e}")
        state["jd_priorities"] = "Error during RAG processing."
        return state

    prompt = ChatPromptTemplate.from_template(
        """You are a JD Alignment specialist. Based on the provided job description context, identify and list the top 5-7 key priorities.

        Context (from RAG):
        {context}

        Question:
        {question}

        Key Priorities:"""
    )
    
    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
        | prompt
        | genai_llm
    )
    response = rag_chain.invoke({"question": "What are the most important requirements for this job?"})
    state["jd_priorities"] = response.content
    return state

def resume_rewriter_agent(state: ResumeOptimizerState):
    """Agent 3: Generates the final tailored resume."""
    state['workflow_log'].append("▶️ Agent 3: Resume Rewriter - Generating tailored resume...")
    prompt = ChatPromptTemplate.from_template(
        """You are an expert career coach. Rewrite the original resume to be perfectly tailored for the given job priorities.

        **Job Description Priorities (from JD Alignment Agent):**
        {jd_priorities}
        **Analysis of Original Resume (from Resume Analyzer Agent):**
        {resume_analysis}
        **Full Original Resume Text:**
        {base_resume}

        Produce a complete, rewritten resume in clean, professional Markdown format.
        **Rewritten Resume:**"""
    )
    chain = prompt | genai_llm
    optimized_resume_content = chain.invoke({
        "jd_priorities": state["jd_priorities"],
        "resume_analysis": state["resume_analysis"],
        "base_resume": state["base_resume_text"]
    })
    state["optimized_resume"] = optimized_resume_content.content
    state['workflow_log'].append("✅ Workflow Complete.")
    return state

def response_tracker_analysis_agent():
    """
    Agent 4's core logic: Analyzes successful resumes to find winning patterns.
    This function is triggered by a button press in the UI.
    """
    st.session_state.tracker_insights = "" # Clear previous insights
    
    successful_resumes = []
    for data in st.session_state.generated_resumes.values():
        if data.get("feedback") == "Yes!":
            successful_resumes.append(data['resume'])
            
    if not successful_resumes:
        st.warning("No successful resumes marked with 'Yes!' to analyze.")
        return

    # Combine successful resumes into one context document
    context = "\n\n---\n\n".join(successful_resumes)
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert career strategist. You have been given one or more resume versions that successfully landed an interview. 
        Your task is to analyze these resumes and identify the common themes, powerful phrases, or effective sections that likely made them successful.

        **Successful Resume(s):**
        {successful_resumes}

        **Analysis and Insights:**
        Based on your analysis, what are the "winning" elements? Please provide a bulleted list of actionable insights and flag the most effective sections for reuse."""
    )
    
    chain = prompt | genai_llm
    with st.spinner("Response Tracker Agent is analyzing successful versions..."):
        insights = chain.invoke({"successful_resumes": context})
        st.session_state.tracker_insights = insights.content


# --- 4. LANGGRAPH GRAPH CONSTRUCTION WITH A DECISION LOOP ---

def check_analysis_quality(state: ResumeOptimizerState):
    """This is the decision-making node. It directs the workflow."""
    if state["review_decision"] == "YES":
        state['workflow_log'].append("↪️ Decision: Analysis approved. Proceeding to next step.")
        return "continue"
    else:
        state['workflow_log'].append("↪️ Decision: Analysis is poor. Looping back to retry.")
        if state['retry_count'] > 1: # Limit retries to 1 to avoid getting stuck
            state['workflow_log'].append("⚠️ Max retries reached. Forcing continuation.")
            return "continue"
        return "retry"

workflow = StateGraph(ResumeOptimizerState)

workflow.add_node("resume_analyzer", resume_content_analyzer_agent)
workflow.add_node("analysis_reviewer", analysis_reviewer_agent)
workflow.add_node("jd_aligner", jd_alignment_agent)
workflow.add_node("resume_rewriter", resume_rewriter_agent)

workflow.set_entry_point("resume_analyzer")
workflow.add_edge("resume_analyzer", "analysis_reviewer")
workflow.add_conditional_edges(
    "analysis_reviewer",
    check_analysis_quality,
    {
        "continue": "jd_aligner",
        "retry": "resume_analyzer"
    }
)
workflow.add_edge("jd_aligner", "resume_rewriter")
workflow.add_edge("resume_rewriter", END)
app = workflow.compile()


# --- 5. STREAMLIT UI ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"🔴 Error reading PDF file: {pdf.name}. Details: {e}")
            return None
    return text

st.set_page_config(page_title="📄 Resume Optimizer (True Agentic)", layout="wide")

if "generated_resumes" not in st.session_state:
    st.session_state.generated_resumes = {}
if "tracker_insights" not in st.session_state:
    st.session_state.tracker_insights = ""


st.header("📄 Resume Optimizer (True Agentic Version)")
st.markdown("This version uses a **LangGraph** multi-agent system with a **decision-making loop** to prove true agentic behavior.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Inputs")
    uploaded_files = st.file_uploader("1. Upload Your Base Resume (PDF)", type="pdf", accept_multiple_files=True)
    job_description = st.text_area("2. Paste the Job Description Here", height=250, placeholder="Paste the full job description here...")

    if st.button("✨ Optimize My Resume", type="primary"):
        if uploaded_files and job_description:
            with st.spinner("The AI Agent team is working..."):
                base_resume_text = get_pdf_text(uploaded_files)
                if base_resume_text:
                    initial_state = {
                        "base_resume_text": base_resume_text,
                        "job_description": job_description,
                        "workflow_log": ["▶️ Workflow initiated..."],
                        "retry_count": 0,
                    }
                    
                    log_placeholder = col2.empty()
                    resume_placeholder = col2.empty()
                    
                    final_state = {}
                    for event in app.stream(initial_state):
                        if "__end__" not in event:
                            agent_name = list(event.keys())[0]
                            final_state = event[agent_name]
                            
                            log_placeholder.empty()
                            with log_placeholder.container():
                                st.subheader("Agentic Process Log")
                                log_container = st.container(height=250)
                                with log_container:
                                    for log in final_state.get('workflow_log', []):
                                        st.text(log)
                    
                    optimized_resume = final_state.get('optimized_resume', 'Error: No resume generated.')
                    
                    resume_placeholder.empty()
                    with resume_placeholder.container():
                        st.subheader("🚀 Your Latest Optimized Resume")
                        st.markdown(optimized_resume)
                    
                    resume_id = str(uuid.uuid4())
                    st.session_state.generated_resumes[resume_id] = {
                        "resume": optimized_resume,
                        "feedback": "Not yet"
                    }
                    st.success("Optimization Complete!")
        else:
            st.warning("⚠️ Please upload a resume and paste a job description.")

with col2:
    if "latest_resume" not in st.session_state or not st.session_state.get("latest_resume"):
        st.subheader("Agentic Process Log")
        st.info("Logs from the agent workflow will appear here in real-time.")
        st.subheader("🚀 Your Latest Optimized Resume")
        st.info("Your optimized resume will appear here after running the analysis.")

st.write("---")
st.header("📊 Agent 4: Response Tracker")
st.markdown("Monitor which resume versions receive more callbacks and **analyze their performance** to flag effective sections for reuse.")

# --- Upgraded Response Tracker UI ---
if not st.session_state.generated_resumes:
    st.info("No resumes generated yet. Run the optimizer to see tracked versions here.")
else:
    st.write("**Tracked Versions:**")
    for resume_id, data in st.session_state.generated_resumes.items():
        with st.expander(f"Version ID: {resume_id[:8]}"):
            st.markdown(data['resume'])
            feedback = st.radio(
                "Did this version get an interview?",
                ("Not yet", "Yes!", "No"),
                key=f"feedback_{resume_id}",
                index=0 if data.get('feedback', 'Not yet') == 'Not yet' else 1 if data.get('feedback') == 'Yes!' else 2,
                horizontal=True,
            )
            st.session_state.generated_resumes[resume_id]['feedback'] = feedback
    
    st.write("---")
    st.write("**Performance Analysis:**")
    if st.button("🚀 Analyze Performance of Successful Resumes"):
        response_tracker_analysis_agent() # Call the new agent function
    
    if st.session_state.tracker_insights:
        st.success("Analysis Complete! Here are the insights:")
        st.markdown(st.session_state.tracker_insights)

