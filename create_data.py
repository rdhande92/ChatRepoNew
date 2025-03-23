import json
from langchain_community.document_loaders import GithubFileLoader
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()
GIT_ACCESS_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
persist_directory = "./ChromaDB"

fetched_formats = ('.cfg', '.ini', '.json', '.md', '.py', '.toml', '.yml', '.yaml')

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""])

# Python Splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=200)

print("All Splitters created...!")

all_chunks = []
qa_chunks = []  # Initialize the Question:Answer chunks list
for each_format in fetched_formats:
    print("Loading from GitHub:", each_format)
    # Initialize the loader
    documents = GithubFileLoader(
        repo="vanna-ai/vanna",
        branch="main",
        access_token=GIT_ACCESS_TOKEN,
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(each_format)
    ).load()

    print("Documents loaded from GitHub for format:", each_format)
    for document in documents:
        if each_format in ('.cfg', '.ini', '.md', '.toml', '.yml', '.yaml'):
            chunks = text_splitter.split_text(document.page_content or "")  # Ensure the content is a string
        elif each_format == '.py':
            chunks = python_splitter.split_text(document.page_content or "")  # Ensure the content is a string
        elif each_format == '.json':
            # Load JSON content from the document
            json_content = json.loads(document.page_content or "")
            metadata = document.metadata  # Retrieve metadata for each document
            for qa_pair in json_content:
                if isinstance(qa_pair, dict):  # Ensure it's a dictionary (question-answer pair)
                    question = qa_pair.get("question", "").strip()
                    answer = qa_pair.get("answer", "").strip()

                    # Avoid empty entries
                    if question and answer:
                        # Create a chunk for the Question:Answer pair with its metadata
                        qa_chunk = {
                            "content": f"Q: {question}\nA: {answer}",
                            "metadata": metadata
                        }
                        qa_chunks.append(qa_chunk)  # Add to the Question:Answer chunks list

        # Validate and add chunks safely
        all_chunks.extend(
            {"content": chunk if isinstance(chunk, str) else str(chunk), "metadata": document.metadata}
            for chunk in chunks
        )

# Append the QA chunks to all_chunks
all_chunks.extend(qa_chunks)

print("All chunks (including Q&A chunks) prepared.")

# Initialize OpenAI Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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