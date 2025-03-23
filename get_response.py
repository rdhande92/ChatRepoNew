from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Specify the directory containing the saved ChromaDB
persist_directory = "./ChromaDB"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Reload the Chroma vector store
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Chroma(
    collection_name="embeddings",  
    embedding_function=embedding_model,
    persist_directory=persist_directory)

# Initialize ChatOpenAI (LLM)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_response_llm_with_memory(question):
    """
    Use the vector store to retrieve context, maintain chat memory, 
    and return an LLM response.
    """
    # Retrieve context from vector store
    context = vector_store.similarity_search(question, k=10)  # Retrieve top 10 most similar chunks

    # Format the context into a readable string for the prompt
    context_string = "\n\n".join([chunk.page_content for chunk in context])

    # Construct conversation
    messages = memory.chat_memory.messages  # Load past interactions from memory
    messages.append(HumanMessage(content=f"User's Question: {question}"))
    messages.append(AIMessage(content=f"Relevant Context: {context_string}"))

    # Construct the prompt with memory and context
    prompt = f"""You are a helpful assistant specializing in coding knowledge and problem-solving.
    The user has asked the following question in an ongoing conversation. 
    Use the memory of our conversation and the provided context to give the best possible detailed response. 
    Respond step-by-step and logically. If you do not find enough information in the context, 
    reply with 'Out-of-scope question.' Do not invent knowledge outside the context.
    User Question: {question}
    Context: {context_string}
    Conversation Memory:
    {messages}
    """

    # Generate response
    response = llm.invoke(prompt)

    # Save conversation history in memory
    memory.chat_memory.add_message(HumanMessage(content=question))
    memory.chat_memory.add_message(AIMessage(content=response.content))

    return response.content

# Example interaction
print(get_response_llm_with_memory("What are the names of all the Cities in Canada?"))

# Example follow-up interaction using memory
print(get_response_llm_with_memory("Which type of query it provide?"))