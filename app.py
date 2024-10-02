import time
import streamlit as st
import re
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="WellMind Companion",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper Functions
def initialize_session_state():
    """Initialize chat history in session state."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

def load_and_split_documents(txt_path, chunk_size=1000):
    """Load and split documents into chunks."""
    loader = TextLoader(txt_path, encoding='utf-8')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    return docs

def setup_vectorstore_and_retriever(docs, embedding_model, k=10):
    """Setup vectorstore and retriever for document search."""
    vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model=embedding_model))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

def initialize_llm(model_name, temperature=0.8, max_tokens=None, timeout=None):
    """Initialize the language model for answering queries."""
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

def get_prompt_with_history(history, user_input):
    """Generate conversation history along with new user input."""
    history_text = "".join([f"Human: {item['user_input']}\n{item['response']}\n" for item in history])
    history_text += f"Human: {user_input}\n"
    return history_text

def create_rag_chain(retriever, llm, system_prompt, query):
    """Create a Retrieval-Augmented Generation (RAG) chain."""
    query = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", get_prompt_with_history(st.session_state.chat_history, query))
    ])
    question_answer_chain = create_stuff_documents_chain(llm, query)
    return create_retrieval_chain(retriever, question_answer_chain)

def handle_query(query, retriever, llm, system_prompt):
    """Process the user query and return the response."""
    rag_chain = create_rag_chain(retriever, llm, system_prompt, query)
    response = rag_chain.invoke({"input": query})['answer']
    st.session_state.chat_history.append({"user_input": query, "response": response})
    return response

# UI Functions
def render_message(content, role):
    """Render user or assistant messages with appropriate styles."""
    code_block_pattern = r'```(python|json)(.*?)```'
    parts = re.split(code_block_pattern, content, flags=re.DOTALL)

    for i in range(0, len(parts), 3):
        text_part = parts[i].strip()
        if text_part:
            if role == "assistant":
                st.markdown(f'<div class="assistant-message">{text_part}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-message">{text_part}</div>', unsafe_allow_html=True)

        if i + 1 < len(parts):
            language = parts[i + 1]  # Language of the code block
            code_part = parts[i + 2].strip()  # Code content
            if code_part:
                st.code(code_part, language=language)

def chat():
    """Main chat function to handle interactions."""
    initialize_session_state()

    # Load and prepare documents
    docs = load_and_split_documents(r"mental_health.txt")
    retriever = setup_vectorstore_and_retriever(docs, embedding_model="models/embedding-001")

    # Initialize the language model
    llm = initialize_llm(model_name="gemini-1.5-flash")

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .assistant-message {
            background-color: #f5f5f5;
            color: #333;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 10px;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
            width: fit-content;
            max-width: 80%;
            text-align: left;
        }
        .user-message {
            background-color: #007BFF;
            color: white;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 10px;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
            width: fit-content;
            max-width: 80%;
            text-align: right;
            margin-left: auto;
        }
        .stTextInput > div > div {
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        render_message(message["content"], message["role"])

    # Input field for new queries
    query = st.chat_input("What's on your mind?")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        render_message(query, "user")

        # Define system prompt for the assistant
        system_prompt = (
            "You are an AI assistant providing empathetic mental health and emotional support to students. "
            "Your role is to listen, understand, and offer comforting, positive guidance. Ensure your responses "
            "are gentle and encouraging. If you don’t know something or if the situation seems severe, suggest that "
            "the user speak to a counselor or a trusted person.\n\n"
            "{context}"
        )

        # Process the query and generate response
        response_text = handle_query(query, retriever, llm, system_prompt)

        # Display assistant response with typing effect
        partial_response = ""
        response_container = st.empty()
        for word in response_text.split():
            partial_response += word + " "
            if "```python" in partial_response:
                code_content = partial_response.split("```python")[1].split("```")[0]
                response_container.code(code_content, language='python')
            else:
                response_container.markdown(f'<div class="assistant-message">{partial_response}</div>', unsafe_allow_html=True)
            time.sleep(0.05)  # Delay for typing effect

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        render_message(response_text, "assistant")

        # Rerun the app to keep UI updated
        st.rerun()

# Run the chat function
chat()