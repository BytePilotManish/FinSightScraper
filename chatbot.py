import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Tuple
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

class FinanceChatbot:
    def __init__(self, domain: str = "finance"):
        self.domain = domain
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.texts = None
        self.load_data()

    def load_data(self):
        """Load the FAISS index and text data."""
        try:
            # Load the FAISS index
            self.index = faiss.read_index(f"data/{self.domain}_index.faiss")
            
            # Load the text data
            with open(f"data/{self.domain}_texts.json", "r", encoding="utf-8") as f:
                self.texts = json.load(f)
            
            print(f"[✓] Loaded {len(self.texts)} documents")
        except Exception as e:
            print(f"[!] Error loading data: {e}")
            raise

    def find_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Find the most relevant documents for a given query."""
        # Encode the query
        query_embedding = self.model.encode([query])[0]
        
        # Search the index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        # Get the relevant documents
        relevant_docs = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.texts):  # Ensure index is valid
                doc = self.texts[idx].copy()
                # Convert distance to similarity score (0 to 1)
                doc['relevance_score'] = float(1 / (1 + distance))
                relevant_docs.append(doc)
        
        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_docs

    def format_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Format the response using the relevant documents."""
        if not relevant_docs:
            return "I'm sorry, I don't have enough information to answer that question."

        # Filter out low relevance documents
        relevant_docs = [doc for doc in relevant_docs if doc['relevance_score'] > 0.3]
        
        if not relevant_docs:
            return "I'm sorry, I couldn't find any relevant information for your question."

        # Start with a general response
        response = f"Here's what I found about your query: '{query}'\n\n"
        
        # Group documents by source
        sources = {}
        for doc in relevant_docs:
            source = doc.get('url', 'Unknown Source').split('/')[2]  # Get domain
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)

        # Add information from each source
        for i, (source, docs) in enumerate(sources.items(), 1):
            response += f"{i}. From {source}:\n"
            
            # Combine content from the same source
            combined_content = []
            for doc in docs:
                if doc['text'] not in combined_content:
                    combined_content.append(doc['text'])
            
            # Add the combined content
            response += "\n".join(combined_content)
            response += "\n\n"

        return response

    def clean_question(self, question: str) -> str:
        """Clean and preprocess the question."""
        # Remove extra whitespace
        question = re.sub(r'\s+', ' ', question)
        # Remove special characters but keep basic punctuation
        question = re.sub(r'[^\w\s.,!?-]', '', question)
        return question.strip()

    def chat(self, question: str) -> str:
        """Main chat method to process questions and return answers."""
        # Clean the question
        question = self.clean_question(question)
        
        # Find relevant documents
        relevant_docs = self.find_relevant_documents(question)
        
        # Format and return the response
        return self.format_response(question, relevant_docs)

def main():
    # Initialize the chatbot
    chatbot = FinanceChatbot()
    
    print("\n=== Finance Chatbot ===")
    print("Type 'quit' to exit")
    print("Ask me anything about finance!\n")
    
    while True:
        # Get user input
        question = input("You: ").strip()
        
        # Check if user wants to quit
        if question.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! Have a great day!")
            break
        
        # Get and print response
        response = chatbot.chat(question)
        print("\nBot:", response, "\n")

if __name__ == "__main__":
    main() 