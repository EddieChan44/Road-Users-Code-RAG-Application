# chat.py
import os
import pickle
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

class RoadUsersCodeChat:
    def __init__(self, history_file="road_users_chat_history.pkl"):
        chroma_directory = "./chroma_db/road_users_code_rag"
        
        # Check if the vector database exists
        if not os.path.exists(chroma_directory):
            raise FileNotFoundError(f"Error: Vector database not found at {chroma_directory}. Please run the training script first.")
        
        # Load models
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        self.vector_db = Chroma(
            persist_directory=chroma_directory,
            collection_name="road_users_code_rag",
            embedding_function=self.embeddings
        )
        
        # Set up LLM
        local_model = "mistral"
        self.llm = ChatOllama(model=local_model)
        
        # Define query prompt for retrieving multiple perspectives
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are an AI language model assistant. "
                "Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. "
                "By generating multiple perspectives on the user question, your goal is to help the user overcome some of the distance-based similarity search limitations. "
                "Provide these alternative questions separated by newlines. Original question: {question}"
            )
        )
        
        # Create retriever
        self.retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),
            self.llm,
            prompt=QUERY_PROMPT
        )
        
        # Define the QA prompt
        template = (
            "Answer the question based ONLY on the following context:\n"
            "{context}\n\n"
            "Conversation History:\n"
            "{history}\n\n"
            "Question: {question}\n\n"
            "Answer the question using the provided context. If the answer is not in the context, politely state that you don't have that information about the Road Users Code."
        )
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Initialize conversation history
        self.history = []
        self.history_file = history_file
        self.load_history()
    
    def load_history(self):
        """Load conversation history if it exists."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    self.history = pickle.load(f)
                print(f"Loaded chat history with {len(self.history)} messages")
            except Exception as e:
                print(f"Could not load history: {e}")
                self.history = []
    
    def save_history(self):
        """Save conversation history."""
        with open(self.history_file, 'wb') as f:
            pickle.dump(self.history, f)
    
    def format_history(self):
        formatted = ""
        for entry in self.history[-10:]:  # Use only the last 10 exchanges for context
            formatted += f"{entry['role']}: {entry['content']}\n"
        return formatted
    
    def process_query(self, question):
        # Add the user question to the history
        self.history.append({"role": "Human", "content": question})
        history_text = self.format_history()
        
        # Build the chain for this specific question
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough(), "history": lambda _: history_text}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke(question)
        
        # Add the answer to the history and save
        self.history.append({"role": "AI", "content": response})
        self.save_history()
        
        return response
    
    def display_history(self):
        for entry in self.history:
            role = entry["role"]
            content = entry["content"]
            print(f"\n{role}: {content}")
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
        self.save_history()
        print("Conversation history cleared")
    
    def chat(self):
        print("Road Users Code Chat Assistant - Type 'exit' to end, 'clear' to reset history")
        
        while True:
            user_input = input("\nYour question: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nEnding chat session. Goodbye!")
                self.save_history()
                break
            elif user_input.lower() == 'clear':
                self.clear_history()
                continue
            
            print("\nThinking...")
            try:
                response = self.process_query(user_input)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again with a different question.")

def run_chat():
    chat_instance = RoadUsersCodeChat()
    chat_instance.chat()

if __name__ == '__main__':
    run_chat()