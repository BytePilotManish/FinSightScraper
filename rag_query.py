import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Error initializing model: {str(e)}")
    raise

def load_data(domain="finance"):
    """Load FAISS index and documents with error handling."""
    try:
        # Check if files exist
        index_path = f"data/{domain}_index.faiss"
        texts_path = f"data/{domain}_texts.json"
        
        if not os.path.exists(index_path) or not os.path.exists(texts_path):
            raise FileNotFoundError(f"Required files not found in data directory")
        
        # Load index and documents
        index = faiss.read_index(index_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        if not docs:
            raise ValueError("No documents loaded from JSON file")
            
        return index, docs
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def format_document(doc):
    """Format a document into a readable summary."""
    title = doc.get('title', 'Untitled')
    text = doc.get('text', '')
    # Take first 300 characters of text
    text = text[:300] + "..." if len(text) > 300 else text
    return f"Title: {title}\nContent: {text}\n"

def ask_query(query, k=2, domain="finance"):
    """Process query and return relevant information."""
    try:
        # Load data
        index, docs = load_data(domain)
        
        # Encode query
        query_embedding = model.encode([query]).astype("float32")
        
        # Search index
        D, I = index.search(query_embedding, k)
        
        # Get relevant documents
        relevant_docs = []
        for i in I[0]:
            if i < len(docs):
                doc = docs[i]
                relevant_docs.append(format_document(doc))
        
        if not relevant_docs:
            return "No relevant information found."
        
        # Combine the information
        response = "Here's what I found:\n\n" + "\n---\n".join(relevant_docs)
        return response
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

def main():
    """Main function with improved error handling and user interaction."""
    print("\n🤖 Finance Knowledge Assistant")
    print("Type 'exit' to quit")
    
    while True:
        try:
            query = input("\n💬 Ask a finance question: ").strip()
            
            if not query:
                print("Please enter a question.")
                continue
                
            if query.lower() == "exit":
                print("Goodbye!")
                break
            
            print("\n🔍 Searching...")
            answer = ask_query(query)
            print("\n📘 Answer:\n", answer)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 