import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Build and manage vector store with FAISS for semantic search"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print("\n⏳ Loading embedding model...")
        self.embed_model = SentenceTransformer(model_name)
        self.index       = None
        self.chunks      = []
        self.metadata    = []

    def build_index(self, chunks: list, metadata: list):
        """Encode chunks and build FAISS index"""
        self.chunks   = chunks
        self.metadata = metadata

        print("⏳ Encoding chunks into embeddings...")
        embeddings = self.embed_model.encode(
            chunks, 
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index (Inner Product = cosine after normalization)
        dimension   = embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅ FAISS index built with {self.index.ntotal} vectors (dim={dimension})")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve top-k most relevant chunks for a query"""
        query_vec = self.embed_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    'chunk':    self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'score':    round(float(score), 4)
                })
        return results
