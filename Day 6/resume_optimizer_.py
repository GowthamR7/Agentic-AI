import streamlit as st
import pdfplumber
import pandas as pd
import google.generativeai as genai

# Load API Key from secrets
genai.configure(api_key="AIzaSyBJqlpUCBaP0ZqIPOxrrKk0eRd9xNSRl60")

st.set_page_config(page_title="Resume Version Optimizer", layout="wide")
st.title("📌 Resume Version Optimizer (Powered by Gemini 1.5 Flash)")

# --- Upload Resume & JD ---
resume_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("📝 Paste the Job Description")

target_roles = st.multiselect(
    "🎯 Choose Target Role(s)",
    ["Backend", "Frontend", "ML", "QA", "DevOps", "Data Analyst"]
)

# --- Extract PDF Text ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# --- Call Gemini API ---
def generate_resume_with_gemini(resume_text, jd_text, role):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"""
You are a resume optimization assistant. Tailor the following resume for the "{role}" role based on this job description. 
Emphasize relevant skills, remove irrelevant ones, and rewrite where necessary to improve alignment with the role.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Return the optimized resume only.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Generate Optimized Resumes ---
if st.button("🚀 Optimize Resume") and resume_file and job_description and target_roles:
    resume_text = extract_text_from_pdf(resume_file)
    st.success("✅ Resume extracted successfully!")

    for role in target_roles:
        with st.spinner(f"Optimizing for {role}..."):
            try:
                output = generate_resume_with_gemini(resume_text, job_description, role)
                st.text_area(f"📌 Tailored Resume for {role}", value=output, height=400)
                st.download_button(
                    f"📥 Download {role} Resume",
                    data=output,
                    file_name=f"{role}_resume.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating resume for {role}: {e}")

# --- Application Tracking ---
st.markdown("## 📊 Track Resume Versions & Application Feedback")

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = pd.DataFrame(columns=["Version", "Role", "Company", "Status"])

with st.form("feedback_form"):
    col1, col2, col3, col4 = st.columns(4)
    version = col1.text_input("Version")
    role = col2.selectbox("Role", target_roles)
    company = col3.text_input("Company")
    status = col4.selectbox("Status", ["Pending", "Interview", "Rejected", "Offer"])

    if st.form_submit_button("➕ Add Entry"):
        new_entry = pd.DataFrame([[version, role, company, status]], columns=st.session_state.feedback_log.columns)
        st.session_state.feedback_log = pd.concat([st.session_state.feedback_log, new_entry], ignore_index=True)

st.dataframe(st.session_state.feedback_log)
