# AI Legal Document Assistant

This project is an AI-powered legal document assistant built using Streamlit, Google Generative AI, and Sentence Transformers. It allows users to upload PDF contracts and receive analysis on key clauses, important dates, and risky terms.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

## Setting Up the Virtual Environment

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required packages**:
   ```bash
   pip install -r requirements.py
   ```

5. **Set up environment variables**:
   Create a `.env` file in the root directory of the project and add your Google Generative AI API key:
   ```plaintext
   API_KEY=your_api_key_here
   ```

## Running the Application

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and go to `http://localhost:8501` to access the application.

## Usage

- Upload a PDF contract using the file uploader in the sidebar.
- The application will analyze the document and provide insights on key clauses, important dates, and any risky terms.
- You can interact with the chatbot to ask specific questions about the contract.
