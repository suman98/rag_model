import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleVectorStore:
    """A simple text-based vector store using TF-IDF for similarity search"""
    def __init__(self):
        self.documents = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def add_documents(self, documents: List[str]):
        """Add documents and create TF-IDF vectors"""
        self.documents = documents
        if documents:
            self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar documents to the query"""
        if not self.documents or self.vectorizer is None:
            return []
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if idx < len(self.documents) and similarities[idx] > 0:
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results


class RAGModel:
    def __init__(self, api_key: str = None):
        """Initialize RAG model with Gemini API"""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or arguments")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.vector_store = SimpleVectorStore()
        self.documents = []
        
    def load_csv_data(self, csv_path: str, chunk_size: int = 5):
        """Load and chunk CSV data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Create text chunks from the data
        documents = []
        for idx in range(0, len(df), chunk_size):
            chunk = df.iloc[idx:idx+chunk_size]
            
            # Convert chunk to readable text format
            chunk_text = "Stock Price Data:\n"
            for _, row in chunk.iterrows():
                chunk_text += f"Symbol: {row['symbol']}, Date: {row['date']}, "
                chunk_text += f"Open: ${row['open']}, High: ${row['high']}, "
                chunk_text += f"Low: ${row['low']}, Close: ${row['close']}, "
                chunk_text += f"Volume: {row['volume']}\n"
            
            documents.append(chunk_text.strip())
        
        self.documents = documents
        print(f"Created {len(documents)} document chunks")
        
        # Create vector store from documents
        self.vector_store.add_documents(self.documents)
        print("Vector store created and ready for queries!")
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents for a query using TF-IDF similarity"""
        results = self.vector_store.similarity_search(query, k=k)
        return [doc for doc, _ in results]
    
    def query(self, user_query: str, k: int = 5) -> str:
        """Query the RAG model and get a response"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve(user_query, k=k)
        
        # Create context from retrieved documents
        context = "\n---\n".join(relevant_docs)
        
        # Create prompt for Gemini
        prompt = f"""You are a helpful assistant analyzing stock price data. 
Use the following data chunks to answer the user's question.

Data Context:
{context}

User Question: {user_query}

Please provide a helpful and accurate answer based on the provided data."""
        
        # Generate response using Gemini
        response = self.model.generate_content(prompt)
        return response.text
        
    def interactive_query(self):
        """Start an interactive query session"""
        print("\n" + "="*60)
        print("RAG Model Interactive Query Session")
        print("="*60)
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'help' for available commands\n")
        
        while True:
            try:
                user_input = input("Query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting query session. Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  - Type your question to query the data")
                    print("  - 'exit' or 'quit' to end the session")
                    print("  - 'help' to show this message\n")
                    continue
                
                print("\nGenerating response...")
                response = self.query(user_input)
                print(f"\nResponse:\n{response}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nExiting query session. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again.\n")


def main():
    """Main function to demonstrate RAG model"""
    try:
        # Initialize RAG model
        rag = RAGModel()
        
        # Load data
        csv_path = os.path.join(os.path.dirname(__file__), "data", "one_min_price.csv")
        rag.load_csv_data(csv_path)
        
        # Start interactive query session
        rag.interactive_query()
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please ensure GEMINI_API_KEY is set in .env file")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
