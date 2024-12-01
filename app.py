import os
import time
import pickle
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import fitz
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ.get("API_KEY"))

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the sentence transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app setup
st.title("AI Legal Document Assistant")

# Initialize session state for chat history and document analysis
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""

# Function to extract text from uploaded PDF
@st.cache_data
def extract_text_from_pdf(pdf_file):
    print("Extracting text from PDF...")
    # Open the file as a file-like object in memory
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to split contract text into smaller chunks (e.g., clauses or sentences)
@st.cache_data
def split_text_into_chunks(text, max_length=500):
    print("Splitting text into chunks...")
    # Simple split by period for illustrative purposes; you can adjust this based on your contract format
    chunks = text.split('. ')
    return [chunk.strip() + '.' for chunk in chunks if len(chunk.strip()) > 0]

# Function to detect expiration dates and other important dates using LLM
@st.cache_data
def detect_important_dates_with_llm(text):
    print("Detecting important dates...")
    prompt = f"""
    Please analyze the following contract text and identify all important dates, such as:
    - Expiration dates of the contract,
    - Renewal dates,
    - Payment deadlines,
    - Notice periods,
    - Any other time-sensitive clauses,etc.

    Provide a concise list of dates with a brief explanation of their significance in the contract.

    Contract text: {text}  # Include a snippet of the contract text for context
    """
    response = model.generate_content(prompt)
    print("Important dates detection completed.")
    return response.text

@st.cache_data
def detect_expiration_dates_with_llm(text):
    print("Detecting expiration dates...")
    prompt = f"""
    Please analyze the following contract text and identify the expiration dates.
    - Mention the clause or section in which the date is specified.
    - Provide the date in a standard format (e.g., YYYY-MM-DD) for automation purposes.

    Contract text: {text}
    """
    response = model.generate_content(prompt)
    print("Expiration dates detection completed.")
    return response.text


# Function to flag risky terms and provide detailed reasoning
@st.cache_data
def flag_risky_terms_with_reasoning(text):
    print("Flagging risky terms...")
    risky_terms = ["termination", "indemnity", "liability", "confidentiality", "force majeure", "jurisdiction", "dispute resolution"]
    flagged_clauses = []

    for term in risky_terms:
        if term.lower() in text.lower():
            prompt = f"""
            The following contract text might contain the term '{term}'. 
            - Please summarize why this term might be risky if that term is present in the contract else ignore it.
            - Provide a concise recommendation to mitigate or revise the term if exists otherwise ignore it.
            Contract text: {text}  # Include a snippet of the contract text.
            """
            response = model.generate_content(prompt)
            flagged_clauses.append({
                "term": term,
                "explanation": response.text
            })
    return flagged_clauses

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF contract", type="pdf")

# If the file is uploaded, process the document and display analysis
if uploaded_file is not None:
    print("PDF file uploaded successfully...")
    st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Generate contract chunks after extracting text
    contract_chunks = split_text_into_chunks(st.session_state.pdf_text)

@st.cache_data
def generate_embeddings_and_index():
    print("Generating embeddings and FAISS index...")
    # Check if embeddings and index already exist in session state or file
    if 'contract_embeddings' in st.session_state and 'faiss_index' in st.session_state:
        return  # Skip generation if embeddings and FAISS index are already available
    
    if uploaded_file is not None:
        # Ensure pdf_text is available in session state before accessing it
        if "pdf_text" in st.session_state:
            pdf_text = st.session_state.pdf_text
        else:
            pdf_text = ""  # or handle the case where pdf_text is not available

        contract_chunks = split_text_into_chunks(pdf_text)
        print("Encoding contract chunks into embeddings...")
        contract_embeddings = embedding_model.encode(contract_chunks, convert_to_tensor=True)

        # Convert embeddings to numpy array
        contract_embeddings_np = np.array(contract_embeddings)

        # Set up FAISS index for similarity search
        index = faiss.IndexFlatL2(contract_embeddings_np.shape[1])
        index.add(contract_embeddings_np)

        # Store embeddings and index in session state for later use
        st.session_state['contract_embeddings'] = contract_embeddings_np
        st.session_state['faiss_index'] = index

        # Optionally, save embeddings to a file for persistent storage
        with open('contract_embeddings.pkl', 'wb') as f:
            pickle.dump(contract_embeddings_np, f)
        faiss.write_index(index, 'faiss_index.index')
        print("Embeddings and FAISS index generation completed.")

@st.cache_data
def load_faiss_index():
    print("Loading FAISS index...")
    try:
        # Load the FAISS index
        index = faiss.read_index('faiss_index.index')
        print("FAISS index loaded successfully.")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

# Function to retrieve relevant clauses based on user query
def retrieve_from_embeddings(user_query, contract_chunks, top_k=5):
    print("Retrieving relevant content based on user query...")
    # Use the FAISS index and embeddings from session state
    index = st.session_state.get('faiss_index', None)
    contract_embeddings_np = st.session_state.get('contract_embeddings', None)
    
    if index is None or contract_embeddings_np is None:
        raise ValueError("Embeddings and FAISS index are not available.")
    
    # Proceed with the retrieval as before
    query_embedding = embedding_model.encode([user_query], convert_to_tensor=True)
    D, I = index.search(query_embedding, top_k)  # D: distances, I: indices
    relevant_clauses = [contract_chunks[i] for i in I[0]]
    return relevant_clauses

@st.cache_data
def analyze_document(pdf_text):
    print("Starting document analysis...")
    # Step 1: Identify and classify clauses in the document
    prompt_for_clauses = f"Please analyze the following contract text and identify key clauses (e.g., termination, indemnity, confidentiality, etc.). Please provide a concise summary for each clause:\n\n{pdf_text}"
    response_for_clauses = model.generate_content(prompt_for_clauses)
    clauses_analysis = response_for_clauses.text

    # Step 2: Identify important dates
    important_dates_analysis = detect_important_dates_with_llm(pdf_text)

    # Step 3: Flag risky terms with detailed reasoning
    flagged_clauses = flag_risky_terms_with_reasoning(pdf_text)

    return clauses_analysis, important_dates_analysis, flagged_clauses

# Create Tabs for Document Analysis and Chatbot
tab1, tab2 = st.tabs(["Analysis", "Chatbot"])

with tab1:
    # Document Analysis
    if st.session_state.pdf_text:
        processing_message = st.empty()
        processing_message.write("Analyzing the document for key clauses, important dates, risky terms, and expiration dates...")

        # Show rotating progress spinner while document is being processed
        with st.spinner('Processing... Please wait.'):
            try:
                clauses_analysis, important_dates_analysis, flagged_clauses = analyze_document(st.session_state.pdf_text)
                processing_message.empty()

                # Display results
                if clauses_analysis:
                    with st.expander("## Clauses Summary"):
                        for line in clauses_analysis.split("\n"):
                            if line.strip():
                                st.markdown(f"- {line.strip()}")


                if flagged_clauses:
                    with st.expander("## Risk Analysis"):
                        for flagged in flagged_clauses:
                            st.markdown(f"**{flagged['term'].capitalize()}**: {flagged['explanation'].split('.')[0]}")  # Limit explanation length for ease of reading
                            st.markdown(f"  **Recommendation**: {flagged['explanation'].split('.')[1] if '.' in flagged['explanation'] else 'Revise clause for clarity and fairness.'}")


                if important_dates_analysis:
                    with st.expander("## Important Dates"):
                        st.write(important_dates_analysis)
                print("Document analysis completed.")
                

            except Exception as e:
                print(f"Error during document analysis: {e}")
                st.write(f"Error during document analysis: {e}")
    else:
        st.write("Please upload a PDF contract for analysis.")

with tab2:
    if uploaded_file is not None:
        # Calculate the hash of the uploaded file
        uploaded_file_hash = hash(uploaded_file.read())  # Read the file to calculate hash
        uploaded_file.seek(0)  # Reset file pointer to the beginning

        # Check if the hash is different from the stored hash in session state
        if 'uploaded_file_hash' not in st.session_state or st.session_state['uploaded_file_hash'] != uploaded_file_hash:
            st.session_state['uploaded_file_hash'] = uploaded_file_hash  # Store the new hash

            # Show spinner while generating embeddings
            with st.spinner('Generating embeddings, please wait...'):
                print("Generating Embeddings")
                generate_embeddings_and_index()  # Only called once after upload
        else:
            print("Embeddings already generated for this file.")

        # Chatbot Functionality
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display chat history
        for message in st.session_state["messages"]:
            st.chat_message(message["role"]).write(message["content"])

        # Display analysis results (clauses, important dates, flagged terms) for chatbot context
        analysis_results = ""
        
        if 'clauses_analysis' in locals():
            analysis_results += f"### Clauses Summary:\n{clauses_analysis}\n\n"
        
        if 'important_dates_analysis' in locals():
            analysis_results += f"### Important Dates:\n{important_dates_analysis}\n\n"
        
        if 'flagged_clauses' in locals():
            analysis_results += "### Risk Analysis:\n"
            for flagged in flagged_clauses:
                analysis_results += f"- **{flagged['term'].capitalize()}**: {flagged['explanation'].split('.')[0]}\n"

        # User input for the chatbot
        user_input = st.chat_input("Enter your question")

        if user_input:
            print("User input received:", user_input)  # Print user input
            st.chat_message("user").write(user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # Show spinner while retrieving relevant clauses
            with st.spinner('Reading the document...'):
                relevant_content = retrieve_from_embeddings(user_input, contract_chunks)

            # Combine relevant clauses into a context string
            context = "\n".join(relevant_content)

            # Combine document text, analysis results, and user query for a more informed response
            prompt = f"""
            You are an AI assistant bot for legal contracts.Keep you replies polite and professional. You have the access to following document analysis and contract content:

            Analysis:
            {analysis_results}

            Contract content:
            {context}

            User question: {user_input}
            """

            try:
                response = model.generate_content(prompt)
                bot_reply = response.text
                print("AI response generated.")  # Print after response generation

                # Add AI's response to chat history
                st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
                st.chat_message("assistant").write(bot_reply)
            except Exception as e:
                print("Error during AI response generation:", e)  # Print error if occurs
                st.session_state["messages"].append({"role": "assistant", "content": str(e)})
                st.chat_message("assistant").write(str(e))

    else:
        st.write("Please upload a PDF contract for analysis.")

# Custom CSS to center and space the tabs
st.markdown("""
    <style>
        /* Ensure the tabs are evenly spaced and centered */
        .stTabs [role="tablist"] {
            display: flex;
            justify-content: space-evenly;
        }

        /* Center text in each tab */
        .stTabs [role="tab"] {
            flex-grow: 1;
            text-align: center;
            font-size: 18px !important;
            padding: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 8px;
        }

        /* Styling for active tab */
        .stTabs [aria-selected="true"] {
            font-weight: bold;
            color: rgb(255, 75, 75);  /* Active tab text color */
        }

        /* Slight hover effect without background color change */
        .stTabs [role="tab"]:hover {
            background-color: #f5f5f5;
        }
        /* Fix the user input field at the bottom */
        .stChatInput {
            position: fixed;
            bottom: 0;
            background-color: white;  /* Optional: Set background color */
            z-index: 1000;  /* Ensure it stays above other elements */
            padding: 10px;  /* Optional: Add some padding */
        }
        .stTabs [role="tablist"] {
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)
