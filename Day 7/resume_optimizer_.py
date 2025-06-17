import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI with the API key
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Failed to configure Google AI. Please make sure your GOOGLE_API_KEY is set correctly in the .env file. Error: {e}")
    st.stop()


# --- 2. CORE "AGENT" FUNCTIONS ---

def get_pdf_text(pdf_docs):
    """
    Resume Content Analyzer Agent (Function Version)
    Extracts text content from uploaded PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {pdf.name}. The file might be corrupted or image-based. Details: {e}")
            return None
    return text


def get_text_chunks(text):
    """
    Splits a long string of text into smaller, semantically meaningful chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    JD Alignment Agent (Part 1: Embedding & Storage)
    Generates embeddings for text chunks and stores them in a FAISS vector store.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store. Error: {e}")
        return None


def get_conversational_chain():
    """
    Resume Rewriter Agent (Part 1: Prompt & Chain Setup)
    Defines the prompt template and sets up the LangChain QA chain.
    """
    prompt_template = """
    You are an expert career coach and professional resume writer. Your task is to rewrite the provided resume to be perfectly tailored for a specific job description.

    **Job Description Priorities:**
    {context}

    **Original Resume Text:**
    {question}

    **Instructions:**
    1.  Analyze the **Job Description Priorities** to understand the most critical skills, experiences, and qualifications.
    2.  Carefully review the **Original Resume Text**.
    3.  Rewrite the resume's content to directly address the priorities from the job description.
    4.  Use strong action verbs and keywords from the job description.
    5.  Emphasize relevant skills, projects, and experiences.
    6.  **IMPORTANT:** Do NOT invent new skills or experiences. Only rephrase and reorder existing information.
    7.  The final output should be a complete, rewritten resume in a clean, professional Markdown format.

    **Rewritten Resume:**
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def handle_user_input(user_question, vector_store):
    """
    Orchestrator Function: Connects RAG to the Rewriter and handles the UI for the response.
    """
    try:
        # JD Alignment Agent (Part 2: Retrieval)
        query = "What are the key skills, responsibilities, and qualifications for this role?"
        docs = vector_store.similarity_search(query)
        
        # Resume Rewriter Agent (Part 2: Execution)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Store the response in session state to display it
        st.session_state.optimized_resume = response["output_text"]

    except Exception as e:
        st.error(f"An error occurred while generating the resume. Error: {e}")
        st.session_state.optimized_resume = None


# --- 3. STREAMLIT UI AND CALLBACKS ---

def handle_save():
    """Callback function for the save button."""
    st.session_state.saved = True
    st.toast("✅ Version saved for tracking!")

def handle_feedback():
    """Callback function for the feedback selectbox."""
    feedback = st.session_state.feedback_key
    if feedback != "Not yet":
        st.toast(f"🗣️ Feedback recorded: '{feedback}'")


def main():
    """
    The main function that builds and runs the Streamlit web app.
    """
    st.set_page_config(page_title="📄 Resume Optimizer AI", page_icon="🤖")

    # Initialize session state variables if they don't exist
    if "optimized_resume" not in st.session_state:
        st.session_state.optimized_resume = None
    if "saved" not in st.session_state:
        st.session_state.saved = False

    st.header("📄 Resume Optimizer with Gemini 1.5 & RAG")
    st.markdown("""
    Welcome to the AI-powered Resume Optimizer! 
    1.  **Upload your base resume** (PDF format).
    2.  **Paste the job description** for the role you're targeting.
    3.  Click the button and our AI agents will rewrite your resume to match the job.
    """)

    job_description = st.text_area("Paste the Job Description Here", height=200)
    uploaded_files = st.file_uploader("Upload Your Resume (PDF)", type="pdf", accept_multiple_files=True)

    if st.button("✨ Optimize My Resume"):
        if uploaded_files and job_description:
            with st.spinner("Processing... This may take a moment."):
                raw_text = get_pdf_text(uploaded_files)
                if raw_text:
                    jd_chunks = get_text_chunks(job_description)
                    vector_store = get_vector_store(jd_chunks)
                    if vector_store:
                        handle_user_input(raw_text, vector_store)
                        st.session_state.saved = False # Reset save state for new resume
                        if st.session_state.optimized_resume:
                             st.success("Optimization Complete!")
        else:
            st.warning("Please upload a resume and paste a job description.")

    # Display the optimized resume if it exists in the session state
    if st.session_state.optimized_resume:
        st.write("---")
        st.subheader("🚀 Your Optimized Resume")
        st.markdown(st.session_state.optimized_resume)

        # --- Interactive Response Tracker Agent ---
        st.write("---")
        st.subheader("📊 Response Tracker")
        st.write("Track the performance of this resume version.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # The button's state is now handled by the callback
            st.button("Save this version", on_click=handle_save, disabled=st.session_state.saved)
        
        with col2:
            # The selectbox now has a key and a callback
            st.selectbox(
                "Did this version get an interview?",
                ["Not yet", "Yes!", "No"],
                key="feedback_key",
                on_change=handle_feedback
            )
        
        if st.session_state.saved:
            st.success("This version is now being 'tracked'.")


if __name__ == "__main__":
    main()