# train.py
import os
import pickle
import shutil
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def run_training():
    print("Starting training process: Creating vector database...")

    local_path = "road_users_code_2020_eng.pdf"
    
    # Check if file exists
    if not os.path.exists(local_path):
        print(f"Error: PDF file '{local_path}' does not exist. Please make sure it is in the working directory.")
        return
    
    # Load PDF document
    print("Loading PDF document...")
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
    
    # Pull necessary models
    print("Pulling necessary models from ollama...")
    ollama.pull('nomic-embed-text')
    ollama.pull('mistral')
    
    # Split text into chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # Define paths for saving data
    chroma_directory = "./chroma_db/road_users_code_rag"
    chunks_path = "./road_users_chunks.pkl"
    
    # Remove any existing database to avoid conflicts
    if os.path.exists(chroma_directory):
        print("Existing vector database found. Removing it...")
        shutil.rmtree(chroma_directory)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    
    # Create the vector store
    print("Creating vector database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_directory,
        collection_name="road_users_code_rag"
    )
    
    # Save the vector database to disk
    vector_db.persist()
    
    # Also save the original chunks for reference
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Vector database saved to disk in '{chroma_directory}'")
    print(f"Original document chunks saved to '{chunks_path}'")

if __name__ == '__main__':
    run_training()