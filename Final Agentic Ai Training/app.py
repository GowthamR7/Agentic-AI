from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
import time

# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()

# Initialize Flask App
app = Flask(__name__)

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY file la illa. Unga key ah .env file la podunga.")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.6)
except Exception as e:
    raise e


# --- 2. LANGGRAPH STATE AND AGENT DEFINITIONS ---

class PlacementAnalysisState(TypedDict):
    raw_data_text: str
    structured_data: str
    insights: str
    trend_comparison: str
    recommendations: str
    report: str
    notifications: str
    workflow_log: List[str]

# (Agent functions with corrected prompt variable names)
def data_collector_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 1: Data Collector - Processing all data files...")
    # FIX: Corrected variable name from {raw_data} to {raw_data_text}
    prompt = ChatPromptTemplate.from_template(
        """You are a Data Collector Agent. You have received data from multiple sources (OKRs, Applications, Rejections, Mock Interviews). Your job is to read all this raw text and convert it into a single, clean, structured summary. Combine information for each student logically.

        Raw Data:
        {raw_data_text}
        
        Consolidated Structured Data Summary:"""
    )
    chain = prompt | llm
    response = chain.invoke(state) # Pass the whole state dictionary
    state["structured_data"] = response.content
    return state

def insight_synthesizer_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 2: Insight Synthesizer - Generating key insights...")
    # FIX: Corrected variable name from {data} to {structured_data}
    prompt = ChatPromptTemplate.from_template(
        """You are an Insight Synthesizer Agent. Analyze the structured placement data and generate 3-5 actionable insights about the root causes of placement rejections.

        Structured Placement Data:
        {structured_data}

        Actionable Insights:"""
    )
    chain = prompt | llm
    response = chain.invoke(state)
    state["insights"] = response.content
    return state
    
def trend_comparator_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 3: Trend Comparator (RAG) - Comparing with historical data...")
    try:
        from langchain_core.runnables import RunnablePassthrough
        with open('historical_data.txt', 'r') as f:
            historical_data = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(historical_data)]
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_template(
            """You are a Trend Comparator Agent. Compare the 'Current Placement Insights' with the 'Historical Data Context' from the RAG system. Highlight new trends or recurring problems.

            Historical Data Context: {context}
            Current Placement Insights: {question}

            Trend Comparison:"""
        )
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()}) | prompt | llm
        response = rag_chain.invoke(state["insights"])
        state["trend_comparison"] = response.content
    except Exception as e:
        state["trend_comparison"] = f"Error during RAG comparison: {e}"
    return state
    
def action_recommender_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 4: Action Recommender - Suggesting actions...")
    # FIX: Corrected variable name from {trends} to {trend_comparison}
    prompt = ChatPromptTemplate.from_template(
        """You are an Action Recommender Agent. Based on the insights and trends, suggest 3 concrete actions the college can take.

        Placement Insights: {insights}
        Trend Comparison: {trend_comparison}
        
        Action Recommendations:"""
    )
    chain = prompt | llm
    response = chain.invoke(state)
    state["recommendations"] = response.content
    return state

def report_generator_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 5: Report Generator - Compiling final report...")
    # FIX: Corrected variable name from {trends} to {trend_comparison}
    prompt = ChatPromptTemplate.from_template(
        """You are a Report Generator Agent. Combine all information into a professional Markdown report with these sections: Key Insights Summary, Trend Comparison Analysis, and Actionable Recommendations.

        Information to use:
        - Key Insights: {insights}
        - Trend Comparison: {trend_comparison}
        - Recommendations: {recommendations}
        
        Final Placement Report:"""
    )
    chain = prompt | llm
    response = chain.invoke(state)
    state["report"] = response.content
    return state

def stakeholder_notification_agent(state: PlacementAnalysisState):
    state['workflow_log'].append("▶️ Agent 6: Stakeholder Notifier - Drafting communications...")
    prompt = ChatPromptTemplate.from_template(
        """Based on the final report, draft a concise email to the Placement Head and a short notification for the college portal.

        Report: {report}
        
        Drafted Communications:"""
    )
    chain = prompt | llm
    response = chain.invoke(state)
    state["notifications"] = response.content
    state['workflow_log'].append("✅ Workflow Complete!")
    return state

# --- LANGGRAPH GRAPH CONSTRUCTION ---
workflow = StateGraph(PlacementAnalysisState)
workflow.add_node("data_collector", data_collector_agent)
workflow.add_node("insight_synthesizer", insight_synthesizer_agent)
workflow.add_node("trend_comparator", trend_comparator_agent)
workflow.add_node("action_recommender", action_recommender_agent)
workflow.add_node("report_generator", report_generator_agent)
workflow.add_node("stakeholder_notifier", stakeholder_notification_agent)
workflow.set_entry_point("data_collector")
workflow.add_edge("data_collector", "insight_synthesizer")
workflow.add_edge("insight_synthesizer", "trend_comparator")
workflow.add_edge("trend_comparator", "action_recommender")
workflow.add_edge("action_recommender", "report_generator")
workflow.add_edge("report_generator", "stakeholder_notifier")
workflow.add_edge("stakeholder_notifier", END)
agent_app = workflow.compile()


# --- 3. FLASK API ROUTES ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to run the agentic workflow."""
    data = request.json
    raw_data_text = data.get('raw_data_text', '')

    if not raw_data_text:
        return jsonify({"error": "No data provided."}), 400

    initial_state = { "raw_data_text": raw_data_text, "workflow_log": [] }
    
    final_state = agent_app.invoke(initial_state)

    return jsonify(final_state)


if __name__ == '__main__':
    # Runs the Flask web server
    app.run(debug=True, port=5001)
