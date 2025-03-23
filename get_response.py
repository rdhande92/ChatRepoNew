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
    
    prompt = f"""You are a Q&A bot specialized in answering questions about a GitHub repository, which we have vectorized.
                    You have access to vectorized (indexed) data from the repository.
                    Your task is to answer user questions using only the information contained in the provided context.

                    Instructions:
                    1. If the answer is found within the indexed context, provide it clearly and concisely.
                    2. If you cannot find an answer in the provided context, respond with: "No relevant data in index."
                    3. Do not use any external information or knowledge beyond the provided context.
                    Context: {context}
                    User: {question}"""

    # Use LangChain/LLM to obtain a response
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0,
                     top_p=1,model="gpt-4o")
    response = llm.invoke(prompt)
    return response.content

# print(get_response_llm("Which campaigns have the lowest CPM?"))
# print(get_response_llm("For the keywords with the lowest CPM , how many impressions did they have?"))
# print(get_response_llm("What are the 10 countries with highest government debt in 2020 ?"))
# print(get_response_llm("What is average Fertility Rate measure of Canada in 2002 ?"))
