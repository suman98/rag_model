#!/usr/bin/env python3
"""
Example usage of the RAG Model for stock price data queries
"""

from rag import RAGModel
import os

def main():
    """Example usage of RAG model"""
    
    # Example 1: Initialize the RAG model
    print("="*60)
    print("RAG Model - Stock Price Data Query Example")
    print("="*60)
    
    rag = RAGModel()
    
    # Example 2: Load CSV data
    csv_path = os.path.join(os.path.dirname(__file__), "data", "one_min_price.csv")
    rag.load_csv_data(csv_path)
    
    print("\n" + "="*60)
    print("Ready to answer questions about your data!")
    print("="*60)
    
    # Example queries you can try:
    example_queries = [
        "What was the highest price for INVESTMENT stock?",
        "Show me the price movements from 2:00 PM to 2:10 PM",
        "What was the trading volume at the highest price?",
        "Analyze the price trend throughout the data",
        "When did the close price reach its maximum?"
    ]
    
    print("\nExample questions you can ask:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")
    
    print("\n" + "-"*60)
    print("Starting interactive session...\n")
    
    # Start interactive query session
    rag.interactive_query()


if __name__ == "__main__":
    main()
