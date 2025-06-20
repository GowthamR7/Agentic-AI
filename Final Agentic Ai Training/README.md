🚀 AI-Driven Placement Insight Generator
An agentic AI system designed to autonomously analyze college placement data, identify critical issues, compare trends, and generate actionable reports to improve student outcomes.

Participant Name: Gowtham R

Demo Video Link: https://drive.google.com/file/d/1TdH_I1mHYA-fGsMxnO0LMQb4uXnqD5PD/view?usp=drivesdk

📝 Project Overview & Flow
In today's competitive job market, placement offices are overwhelmed with vast amounts of data from student applications, interviews, and rejections. Manually analyzing this data to find meaningful patterns is nearly impossible.

This project introduces an AI-powered solution: a crew of six specialized AI agents that work collaboratively to solve this problem. The system takes raw placement data (from multiple CSV files), processes it through a sequential agentic workflow, and produces a comprehensive analysis with actionable insights.

The entire workflow is orchestrated using LangGraph, ensuring a robust and transparent process where each agent performs its designated task before passing the results to the next, mimicking a real-world analytics team.

🤖 Agentic Workflow Diagram
The system operates based on a clear, sequential flow of information between six specialist agents. The diagram below illustrates this agentic flow.

```bash
[Input: 4 CSV data files]
           |
           ▼
+---------------------------------+
| 1. Data Collector Agent         |
| (Cleans & Structures Data)      |
+---------------------------------+
           |
           ▼
+---------------------------------+
| 2. Insight Synthesizer Agent    |
| (Finds Key Problems)            |
+---------------------------------+
           |
           ▼
+---------------------------------+
| 3. Trend Comparator Agent (RAG) |
| (Compares with Historical Data) |
+---------------------------------+
           |
           ▼
+---------------------------------+
| 4. Action Recommender Agent     |
| (Suggests Solutions)            |
+---------------------------------+
           |
           ▼
+---------------------------------+
| 5. Report Generator Agent       |
| (Compiles Final Report)         |
+---------------------------------+
           |
           ▼
+---------------------------------+
| 6. Stakeholder Notification Agent|
| (Drafts Emails/Alerts)          |
+---------------------------------+
           |
           ▼
[Output: Final Report & Notifications]

```

✨ Project Showcase
Here are some screenshots of the final application in action, demonstrating the user interface and the detailed, multi-tabbed output.

Main UI with Live Agent Log

Insights & Recommendations

Final Report & Notifications

🛠️ Tech Stack & Architecture
Frontend: Flask & HTML/CSS/JS (A simple, browser-based UI for data upload and results display).

Backend: Flask, Python (A robust server to handle API requests and run the AI workflow).

Agent Framework: LangGraph (To orchestrate the complex, multi-step agent workflow).

LLM Orchestration: LangChain

LLM: Google Gemini 1.5 Flash

Database Strategy:

Primary Data Source: CSV files (okrs.csv, applications.csv, etc.) acting as a flat-file database, uploaded by the user.

Vector Database (for RAG): FAISS, an in-memory vector store, is used to index historical data and enable high-speed semantic search for the Trend Comparator Agent.

⚙️ Setup and Installation (macOS)
Follow these steps to get the project running locally.

1. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep project dependencies isolated.

# Navigate to your project folder
```bash
cd path/to/your/placement-insight-final
```
# Create the virtual environment
```bash
python3 -m venv venv
```

# Activate the virtual environment
```bash
source venv/bin/activate
```

2. Install Dependencies
Create a file named requirements.txt in your project folder with the following content:

```bash
Flask
langgraph
langchain
langchain-community
langchain-google-genai
python-dotenv
pandas
faiss-cpu
```

Now, run the installation command using pip3:

```bash
pip3 install -r requirements.txt
```

3. Set Up Your API Key
Create a file named .env in the root of the project folder.

Get your API key from Google AI Studio.

Add your API key to the .env file like this:

```bash
GOOGLE_API_KEY="AIzaSy...your...secret...key"
```

4. Prepare the Project Files
Ensure you have these files and folders set up correctly:

A data/ folder on your computer containing your four CSV files: okrs.csv, applications.csv, rejections.csv, and mock_interviews.csv. You will upload these from the browser.

A historical_data.txt file in the main project directory.

An assets/ folder containing your three screenshot images (e.g., screenshot_ui.png, screenshot_insights.png, screenshot_report.png).

▶️ How to Run the Application
Once the setup is complete, run the following command from your project folder (with your virtual environment active):

```bash
python3 app.py
```

This will start the Flask web server. Your terminal will show a message like * Running on 
```bash
http://127.0.0.1:5001.

```

Open your web browser (like Chrome or Safari) and go to that address: 
```bash
http://127.0.0.1:5001
```

The web application will load, and you can upload your files and click the "Generate Placement Insights" button to see the agent crew in action!