# WellMind Companion

**WellMind Companion** is an AI-powered chatbot designed to provide mental health and emotional support to students. It interacts with users in a friendly and supportive manner, offering positive guidance based on user input and relevant documents. The project leverages natural language processing (NLP) and Retrieval-Augmented Generation (RAG) to provide users with empathetic and personalized responses.

## Features

- **Empathetic Conversational Agent**: Provides comforting and positive guidance tailored to students' emotional needs.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with generative AI to provide relevant, contextual responses.
- **Customizable Chat History**: Utilizes conversation history for more contextual interactions.
- **Document Search**: Retrieves information from a provided mental health dataset to assist in conversations.
- **Typing Effect**: Simulates a natural typing effect in the user interface.
- **Simple UI**: Easy-to-use interface built with Streamlit for interaction.

## Prerequisites

Before running the project, make sure you have the following installed:

- Python 3.8+
- Google API keys and Chroma for embedding-based document search

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/wellmind-companion.git
   cd wellmind-companion
   ```

2. **Upgrade pip** (to ensure compatibility with the required packages):

   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install dependencies:**

   Use the provided `requirements.txt` to install all necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project directory and add your API keys for the required services (Google Generative AI and Chroma) in the following format:

   ```bash
   GOOGLE_API_KEY="your-google-api-key"
   ```
   Replace `"your-google-api-key"` with your actual Google API key.

## Requirements

The project depends on the following libraries, as listed in `requirements.txt`:

- `streamlit`
- `streamlit-option-menu`
- `streamlit-chat`
- `streamlit-webrtc`
- `streamlit-ace`
- `langchain`
- `langchain-google-genai`

Install all dependencies by following the instructions above.

## Project Structure

- `app.py`: The main Streamlit application file for running the chatbot.
- `mental_health.txt`: A dataset containing mental health-related content that the chatbot retrieves relevant information from.
- `README.md`: Project documentation.
- `.env`: Configuration file containing API keys for authentication.
- `requirements.txt`: A list of dependencies required to run the project.

## Usage

To run the application, use the following command in your terminal:

```bash
streamlit run app.py
```

This will start the Streamlit app, and you can interact with the WellMind Companion through your browser.

### How It Works

1. **Initialization**: The app initializes the chat session and loads mental health-related documents from `mental_health.txt`.
2. **Document Search**: The documents are split into chunks, and a vectorstore is created using Chroma for efficient retrieval.
3. **AI Interaction**: The user inputs queries, and the chatbot generates empathetic responses using the Google Generative AI model. The assistant follows a system prompt designed to provide supportive and empathetic feedback.
4. **Conversation History**: The chatbot keeps track of the chat history to offer contextually aware responses.

## Customization

You can customize various aspects of the chatbot, including:

- **System Prompt**: Adjust the behavior and tone of the assistant by modifying the system prompt in the `handle_query` function.
- **Documents**: Update or replace `mental_health.txt` with your own documents to tailor the chatbot for other purposes.
- **UI Design**: Customize the chat interface by editing the CSS defined in the `st.markdown()` section to suit your preferred design.

## Acknowledgements

This project uses the following technologies:

- [Streamlit](https://streamlit.io/) - For building the interactive UI.
- [Langchain](https://www.langchain.com/) - For integrating language models and document retrieval.
- [Google Generative AI](https://cloud.google.com/ai-generative) - For generating AI-powered responses.
- [Chroma](https://www.trychroma.com/) - For efficient document search.