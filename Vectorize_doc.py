import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("Missing TOGETHER_API_KEY! Please add it to the .env file.")

# Set up logging
log_file_path = os.path.join(os.path.dirname(__file__), 'vectorize_oe.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    logging.info("Initializing embeddings...")

    # Initialize embeddings
    embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

    logging.info("Loading documents from directory...")

    # Load documents from the "Data" directory
    loader = DirectoryLoader(path="Data", glob="*.pdf")
    documents = loader.load()

    if not documents:
        logging.warning("No documents found in the directory.")
    else:
        logging.info(f"Loaded {len(documents)} documents.")

    logging.info("Splitting documents into chunks...")

    # Split documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    text_chunks = text_splitter.split_documents(documents)

    logging.info(f"Generated {len(text_chunks)} text chunks.")

    logging.info("Creating vector database...")

    # Create vector database
    vectordb = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory="vector_db_dir_2"
    )

    logging.info("Vectorization completed successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise
