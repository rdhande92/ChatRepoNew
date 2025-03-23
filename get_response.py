from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
load_dotenv()

# Specify the directory containing the saved ChromaDB
persist_directory = "./ChromaDB"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Reload the Chroma vector store
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Chroma(
    collection_name="embeddings",  
    embedding_function=embedding_model,
    persist_directory=persist_directory     
)

def get_response_llm(question):

    context = vector_store.similarity_search(question, k=10)  
    
    prompt = f"""You are a helpful assistant, you have good knowledge in coding and you will use 
                the provided context to answer user question.
                Read the given context before answering questions and think step by step. 
                If you can not answer a user question based on the provided context, inform the user. 
                Do not use any other information for answering user quesion.
                Context: {context}
                User: {question}
                If you are unable to find the answer in the context, 
                reply with 'Out-of-scope questions."""

    # Use LangChain/LLM to obtain a response
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    return response.content

print(get_response_llm("What are the names of all the Cities in Canada?"))
