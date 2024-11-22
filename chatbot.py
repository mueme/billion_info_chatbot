import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

# Streamlit web UI configuration
st.set_page_config(page_title="High-End AI Chatbot", page_icon="🤖", layout="wide")

# Load environment variables from .env file
load_dotenv()

# Set environment variable for OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI LLM model using GPT-4 turbo
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, temperature=0.4)
chat_model = ChatOpenAI(model_name="gpt-4", streaming=True, openai_api_key=openai_api_key, temperature=0.4)

# Load documents for RAG using SimpleDirectoryLoader
document_loader = DirectoryLoader("./data", glob="**/*.txt")
documents = document_loader.load()

# Split documents into manageable chunks for indexing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_chunks = text_splitter.split_documents(documents)

# Create embeddings from documents and build FAISS index
embeddings = OpenAIEmbeddings()
if len(doc_chunks) > 0:
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
else:
    st.error("No document chunks were created. Please check the documents directory.")

# Define memory for long-term conversations
# Removing memory as we're handling conversation history manually

# Define the prompt for the chatbot with a system message
system_message = """
당신은 매우 상냥하고 따뜻한 빌리언21의 상담원 AI 챗봇입니다. 당신은 항상 답변을 할때마다 매우 활발하고 상냥한 투로 답변하며 항상 유저의 질문에 적극적으로 답변하며 그 다음 질문에 대해 말하며 대화를 아주 유기적으로 이끌어갑니다.
당신은 마치 누군가의 연인(여자친구)처럼 유저를 매우 사랑스럽게 대합니다. 답변을 할때마다 귀여운 이모티콘을 쓰고, 그가 답변의 내용을 이해하였는지 아주 친절하게 접근합니다.
당신은 유저가 요청하지않는한 절대로 거짓말하지 않습니다. 헛소리도 하지 않습니다. 오직 유저만을 생각하며 그에게 정확한 정보를 전달해주길 매우 열성적으로 바로고 있습니다.
"""
prompt = ChatPromptTemplate.from_messages([("system", system_message), ("user", "{user_input}")])

# Construct Retrieval-QA chain for RAG functionality
try:
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    retrieval_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
except Exception as e:
    retrieval_qa_chain = None
    st.error(f"An error occurred during the initialization of the vectorstore or QA chain: {str(e)}")

st.title("🤖 Billion info bot")
st.markdown("빌리언21의 AI 챗봇")

# Streamlit UI components for conversation
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Display the conversation history with a modern chat-like UI
st.markdown("---")
st.header("대화기록")
conversation_container = st.container()
for user_message, bot_response in st.session_state['conversation']:
    with conversation_container:
        st.markdown(f"<div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>"
                    f"<div style='background-color: #007bff; color: white; padding: 10px; border-radius: 15px 15px 0 15px; max-width: 70%;'>"
                    f"{user_message}"
                    f"</div>"
                    f"</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>"
                    f"<div style='background-color: #f1f0f0; color: black; padding: 10px; border-radius: 15px 15px 15px 0; max-width: 70%;'>"
                    f"{bot_response}"
                    f"</div>"
                    f"</div>", unsafe_allow_html=True)

# Scroll to the bottom after each new message
st.markdown("""<script>var conversationDiv = document.getElementsByClassName('streamlit-container');
             if (conversationDiv.length > 0) {
                 conversationDiv[0].scrollIntoView({ behavior: 'smooth', block: 'end' });
             }</script>""", unsafe_allow_html=True)

# Chat input below the conversation history
st.markdown("---")
user_input = st.text_input("💬 Enter your message here:", placeholder="Type your message...", key='user_input')
if user_input and retrieval_qa_chain:
    # Run the input through the retrieval QA chain to get the response
    with st.spinner("Thinking..."):
        # Reinforce the system message in every query
        full_conversation = '\n'.join([f"You: {u}\nAI: {a}" for u, a in st.session_state['conversation']])
        query_with_history = f"{system_message}\n{full_conversation}\nYou: {user_input}"
        response = retrieval_qa_chain.run({"query": query_with_history})
        st.session_state['conversation'].append((user_input, response))
    del st.session_state['user_input']
    st.rerun()
elif not retrieval_qa_chain:
    st.error("Retrieval QA Chain is not properly initialized. Please check the document loading, embeddings, or indexing process.")

# Allow for clearing the conversation
st.markdown("---")
if st.button("🗑️ Clear Chat"):
    st.session_state['conversation'] = []
    st.rerun()
