from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_core.messages import HumanMessage
from datetime import datetime

load_dotenv()
connection_string='mongodb+srv://user:pwd@cluster.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0' 
import pymongo


client=pymongo.MongoClient(connection_string)
db = client.myfirstDatabase
my_session_collection = db["chatbot_rag_sessions"]

llm=ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'],model='llama3-8b-8192')
loader=PyMuPDFLoader('redshift-gsg.pdf')
docs=loader.load()
chunking=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=chunking.split_documents(docs)
chunks
embeddings=OllamaEmbeddings(model='nomic-embed-text')
db=FAISS.from_documents(chunks,embeddings)


def get_session(session_id:str)-> BaseChatMessageHistory:
    session=my_session_collection.find_one({"session_id":session_id})
    if not session:
        # Create new session
        new_session = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "chat_history": [],
            "summary": "",
            "updated_at": datetime.now()
        }
        my_session_collection.insert_one(new_session)
        return new_session
    return session

def update_session(session_id, chat_history, summary):
    my_session_collection.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "user_message": chat_history,
                "summary": summary,
                "updated_at": datetime.now()
            }
        }
    )

def summarize_chat(chat_history):
    
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # Using LangChain's summarization
    summary_chain = ConversationSummaryMemory(llm=llm)
    print(f'summary_chain:{summary_chain}')
    summary = summary_chain.predict_new_summary(chat_history, conversation)
    return summary

def process_query(session_id, user_query):
    # Get or create session
    session = get_session(session_id)
    print(session)
    chat_history = session.get("chat_history", [])
    print(chat_history)
    previous_summary = session.get("summary", "")
    print(previous_summary)
    MAX_CHAT_HISTORY=3
    # Initialize RAG chain
    retriever = db.as_retriever()
    
    
    # Create memory based on current state
    if len(chat_history) >= MAX_CHAT_HISTORY:
        # Summarize if history is too long
        summary = summarize_chat(chat_history)
        memory = ConversationSummaryMemory(
            llm=llm,
            buffer=summary + "\n" + previous_summary
        )
        print(memory)
        # Reset chat history but keep summary
        chat_history = []
    else:
        memory = ConversationBufferMemory()
        for msg in chat_history:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    
    # Get response
    response = qa_chain.run(user_query)
    
    # Update chat history
    chat_history.extend([
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": response}
    ])
    
    # Get current summary (if any)
    current_summary = ""
    if isinstance(memory, ConversationSummaryMemory):
        current_summary = memory.buffer
    
    # Update session in MongoDB
    update_session(session_id, chat_history, current_summary)
    
    return response

# Example usage
if __name__ == "__main__":
    session_id = "test_session_123"  # In real app, generate or get from user
    
    
    user_input = input("Hi I am Divya.How to create user in redshift cluster?")
    
    response = process_query(session_id, user_input)
    print(f"AI response: {response}")



