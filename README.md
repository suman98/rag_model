# RAG Model - Stock Price Data Query System

A Retrieval-Augmented Generation (RAG) system that allows you to query stock price data using the Gemini API.

## Features

- **Data Loading**: Reads stock price data from CSV files
- **Embedding Generation**: Creates vector embeddings for data chunks using Google's embedding model
- **Similarity Search**: Finds relevant data chunks based on your queries
- **Intelligent Responses**: Uses Gemini API to generate contextual answers based on retrieved data
- **Interactive Query**: Chat-like interface to query your data

## Setup Instructions

### 1. Get a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Click "Create API key"
3. Copy your API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Edit the `.env` file and replace `your_api_key_here` with your actual Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

## Usage

### Run Interactive Query Session

```bash
python rag.py
```

Then type your questions about the stock price data:

```
Query: What was the highest price for INVESTMENT stock on 2025-01-02?
Query: Show me the trading volume trends
Query: Analyze the price movement between 2:00 PM and 2:20 PM
```

### Programmatic Usage

```python
from rag import RAGModel

# Initialize the RAG model
rag = RAGModel()

# Load your CSV data
rag.load_csv_data("data/one_min_price.csv")

# Query the data
response = rag.query("What was the closing price at 14:15?")
print(response)

# Or use interactive mode
rag.interactive_query()
```

## Data Format

The system expects CSV files with the following columns:
- `id`: Unique identifier
- `symbol`: Stock symbol
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `interval`: Time interval (in minutes)
- `date`: Date and time
- `volume`: Trading volume

## How It Works

1. **Data Loading**: Reads the CSV and chunks the data into smaller pieces for better embeddings
2. **Embedding**: Converts each data chunk into a vector using Google's embedding model
3. **Retrieval**: When you query, it finds the most similar data chunks based on semantic similarity
4. **Generation**: Passes the retrieved context to Gemini API to generate a natural language response
5. **Display**: Returns the response in a readable format

## Example Queries

- "What was the average closing price for the INVESTMENT stock?"
- "Show me periods with high trading volume"
- "When did the price drop below 98.20?"
- "Analyze the price trend from 14:12 to 14:20"

## Notes

- The system uses cosine similarity for vector search
- Data is chunked into groups of 5 rows by default (adjustable)
- Embeddings are generated using Google's `embedding-001` model
- Responses are generated using the `gemini-1.5-flash` model

## Troubleshooting

**"GEMINI_API_KEY not found"**: Make sure your `.env` file exists and contains your API key

**Rate limiting**: If you get rate limit errors, the system will retry. Consider adding delays for large datasets.

**Memory issues**: For very large datasets, consider processing in batches or using a more sophisticated vector database like FAISS

## Future Enhancements

- [ ] SQL-based querying support
- [ ] FAISS integration for faster similarity search
- [ ] Multi-file support
- [ ] Data export/visualization
- [ ] Conversation history
- [ ] Custom embedding models
