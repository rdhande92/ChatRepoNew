import json
from langchain_community.document_loaders import GithubFileLoader
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma


# Load environment variables from .env
load_dotenv()
GIT_CCESS_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

fetched_formats = ('.cfg', '.ini', '.json', '.md', '.py', '.toml', '.yml', '.yaml')
# fetched_formats = ('.cfg', '.toml')

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
# Python Splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
)

print("All Splitters created...!")

all_chunks = []

for each_format in fetched_formats:
    print("Loading from GitHub:", each_format)
    # Initialize the loader
    documents = GithubFileLoader(
        repo="vanna-ai/vanna",
        branch="main",
        access_token=GIT_CCESS_TOKEN,
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(each_format)
    ).load()

    print("Documents loaded from GitHub for format:", each_format)
    for document in documents:
        if each_format in ('.cfg', '.ini', '.md', '.toml', '.yml', '.yaml', '.json'):
            chunks = text_splitter.split_text(document.page_content or "")  # Ensure the content is a string
        elif each_format == '.py':
            chunks = python_splitter.split_text(document.page_content or "")  # Ensure the content is a string

        # Validate and add chunks safely
        all_chunks.extend(
            {"content": chunk if isinstance(chunk, str) else str(chunk), "metadata": document.metadata}
            for chunk in chunks
        )

# Initialize OpenAI Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

persist_directory = "./ChromaDB"

# Create a Chroma vector store
vectorstore = Chroma(collection_name="embeddings", embedding_function=embedding_model, 
                     persist_directory=persist_directory)

# Embed and store chunks in ChromaDB
for chunk in all_chunks:
    # Ensure text is a string before embedding
    chunk_content = chunk["content"]
    if not isinstance(chunk_content, str):  # Validate that the chunk is a string
        chunk_content = str(chunk_content)  # Convert to string if necessary

    # Skip empty or non-meaningful text
    if chunk_content.strip():  # Ensure the chunk is not empty after stripping whitespace
        vectorstore.add_texts(
            texts=[chunk_content],          # The content of the chunk
            metadatas=[chunk["metadata"]]  # The metadata associated with the chunk
        )

print("All chunks embedded and stored in ChromaDB successfully!")
print(f"ChromaDB successfully saved to {persist_directory}!")