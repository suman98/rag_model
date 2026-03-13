import os
import pandas as pd
from simple_rag import (
    StockDataProcessor,
    VectorStore,
    StockRAG,
    generate_stock_data
)
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# STEP 5: Main Execution
# ============================================================

def main():
    # ── CONFIG ──────────────────────────────────────────────
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # 🔑 Loaded from .env or OS env
    # ────────────────────────────────────────────────────────
    print(GEMINI_API_KEY)
    # 1. Load Stock Price and Financial Data
    print("📊 STEP 1: Loading Stock Data")
    print("-" * 40)
    
    try:
        stock_prices_df = pd.read_csv('data/stock_prices.csv')
        print(f"✅ Loaded stock prices: {len(stock_prices_df)} rows")
        
        financials_df = pd.read_csv('data/financials_data.csv')
        print(f"✅ Loaded financials data: {len(financials_df)} rows")
    except FileNotFoundError as e:
        print(f"❌ Error loading data files: {e}")
        return

    # 2. Process into text chunks
    print("\n📝 STEP 2: Creating Text Chunks")
    print("-" * 40)
    processor = StockDataProcessor(stock_prices_df, financials_df)
    chunks, metadata = processor.create_text_chunks()

    # 3. Build Vector Store
    print("\n🗄️  STEP 3: Building Vector Store")
    print("-" * 40)
    vector_store = VectorStore()
    vector_store.build_index(chunks, metadata)

    # 4. Initialize RAG
    print("\n🤖 STEP 4: Initializing RAG Pipeline")
    print("-" * 40)
    rag = StockRAG(vector_store, GEMINI_API_KEY)

    # 5. Sample Queries
    print("\n💬 STEP 5: Running Sample Queries")
    print("=" * 60)

    questions = [
        "What was Apple's (AAPL) highest stock price?",
        "Compare the weekly performance of TSLA and MSFT",
        "Which stock had the highest trading volume?",
        "What is the total return for GOOGL over the period?",
        "On which date did AMZN have the biggest price drop?"
    ]

    for question in questions:
        result = rag.query(question, top_k=5)
        print(f"\n✅ ANSWER:\n{result['answer']}")
        print("-" * 60)

    # 6. Interactive Mode
    print("\n🎯 INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)
    while True:
        user_input = input("\n❓ Your question: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if not user_input:
            continue
        result = rag.query(user_input, top_k=5)
        print(f"\n✅ ANSWER:\n{result['answer']}")


if __name__ == "__main__":
    main()
