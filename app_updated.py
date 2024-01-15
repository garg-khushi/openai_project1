import os
import shutil
from git import Repo
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Repo path in local directory
REPO_PATH = "./Cloned_repos"

# Utility Functions
def delete_local_repo(path):
    """Delete a local repository."""
    if os.path.exists(path):
        shutil.rmtree(path)

def create_local_repo(path):
    """Create a local repository directory."""
    os.makedirs(path, exist_ok=True)

def clone_repository(repo_url, to_path):
    """Clone a repository from a given URL."""
    delete_local_repo(to_path)
    create_local_repo(to_path)
    Repo.clone_from(repo_url, to_path=to_path)

def process_documents(path, sub_directory=None):
    """Process documents in the given path."""
    # If sub_directory is not provided, use the base path
    process_path = path if not sub_directory else os.path.join(path, sub_directory)
    
    loader = GenericLoader.from_filesystem(process_path, glob="**/*",
                                        suffixes=[".py", ".java", ".cpp", ".hpp", ".c", ".h", ".js", ".md", ".sh",".bat"],
                                        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500))
    documents = loader.load()

    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                    chunk_size=2000, chunk_overlap=200)
    texts = documents_splitter.split_documents(documents)
    return texts

def setup_vector_db(texts):
    """Setup vector database for document processing."""
    embeddings = OpenAIEmbeddings(disallowed_special=())
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory='./data')
    vectordb.persist()
    return vectordb

def setup_conversational_chain():
    """Setup the conversational retrieval chain."""
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)
    return qa_chain

# Streamlit Interface
st.title("GitHub Repository Code Analyzer")
# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("## Made by: Khushi Garg 👋")
st.sidebar.text("Additional Info 💻 ")
st.sidebar.markdown("[Project code ](https://github.com/garg-khushi/openai)")
st.sidebar.text("Mail-id : khushi-garg123@gmail.com")


repo_url = st.text_input("Enter the Repository URL", "Github Repository url")
sub_directory = st.text_input("Enter the subdirectory (optional)")
question = st.text_input("Enter your question")


if st.button("Analyze"):
    if repo_url and question:
        try:
            clone_repository(repo_url, REPO_PATH)
            texts = process_documents(REPO_PATH, sub_directory)
            vectordb = setup_vector_db(texts)
            qa_chain = setup_conversational_chain()
            result = qa_chain.invoke(question)
            st.write(result['answer'])
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.error("Please enter the repository URL and your question.")
