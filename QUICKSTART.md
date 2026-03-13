# RAG Model - Quick Start Guide

## What Has Been Created

Your RAG (Retrieval-Augmented Generation) system for querying stock price data includes:

### Files Created:
1. **rag.py** - Main RAG system implementation
   - `RAGModel` class: Core RAG functionality
   - `SimpleVectorStore` class: In-memory vector database for embeddings
   - Interactive query interface

2. **example_usage.py** - Example script showing how to use the RAG model

3. **test_setup.py** - Setup verification script

4. **requirements.txt** - Python dependencies

5. **README.md** - Comprehensive documentation

6. **.env** - Environment variables (contains GEMINI_API_KEY)

## Quick Start

### Step 1: Set Your Gemini API Key
Edit `.env` file and replace `your_api_key_here` with your actual API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### Step 2: Install/Verify Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
python test_setup.py  # Verify everything is set up
```

### Step 3: Run the RAG System

**Option A: Run rag.py directly**
```bash
python rag.py
```

**Option B: Run the example script**
```bash
python example_usage.py
```

### Step 4: Ask Questions
Once running, type your questions about the stock price data:

```
Query: What was the highest price for INVESTMENT?
Query: Show me price movements between 14:15 and 14:20
Query: Analyze the trading volume patterns
Query: exit  # To quit
```

## How It Works

1. **Data Loading**: Reads your CSV file and chunks it into manageable pieces
2. **Embeddings**: Converts each data chunk into numerical vectors (embeddings)
3. **Retrieval**: When you query, it finds the most relevant data chunks
4. **Generation**: Uses Gemini API to generate a natural language response based on the retrieved data
5. **Response**: Returns a helpful answer about your data

## Key Features

✓ **Semantic Search** - Understands the meaning of your questions, not just keyword matching
✓ **Context-Aware** - Generates answers based on actual data
✓ **Interactive** - Chat-like interface for asking multiple questions
✓ **Flexible** - Works with any CSV file with similar data structure
✓ **No Extra Infrastructure** - In-memory vector storage (no database needed)

## Example Queries to Try

- "What was the price trend throughout the trading session?"
- "During which time period was the volume highest?"
- "What was the average closing price?"
- "Show me the open and close prices for the entire dataset"
- "Were there any significant price jumps?"

## Customization

You can modify:
- **Chunk size**: Change `chunk_size=5` in `load_csv_data()` (default: 5 rows per chunk)
- **Retrieved documents**: Change `k=5` in `query()` method (default: 5 most relevant chunks)
- **Data columns**: Adjust the text format in `load_csv_data()` method
- **Models**: Change `text-embedding-004` for embeddings or `gemini-2.0-flash` for generation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "GEMINI_API_KEY not found" | Add your API key to `.env` file |
| Import errors | Run `pip install -r requirements.txt` |
| Slow responses | Reduce chunk size or number of retrieved documents |
| Memory issues with large datasets | Implement FAISS (see README Advanced section) |

## Next Steps

- Try different queries and see how the system responds
- Modify the data preprocessing in `load_csv_data()` for custom formats
- Integrate into your own application by importing `RAGModel` class
- Scale to larger datasets using FAISS (see README for details)

## API Keys & Security

⚠️ **Important**: Never commit your `.env` file with the API key to version control
- The `.env` file is already in `.gitignore` (if using git)
- Keep your API key private and secure
- Regenerate keys if accidentally exposed

Happy querying! 🚀
