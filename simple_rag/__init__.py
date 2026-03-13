"""Simple RAG (Retrieval-Augmented Generation) package for stock market analysis"""

from .data_processor import StockDataProcessor
from .vector_store import VectorStore
from .rag import StockRAG
from .utils import generate_stock_data

__all__ = [
    'StockDataProcessor',
    'VectorStore',
    'StockRAG',
    'generate_stock_data'
]
