import google.generativeai as genai


class StockRAG:
    """RAG (Retrieval-Augmented Generation) pipeline for stock market analysis"""

    def __init__(self, vector_store, gemini_api_key: str):
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model        = genai.GenerativeModel('gemini-2.5-flash')
        self.vector_store = vector_store

    def build_prompt(self, query: str, retrieved_chunks: list) -> str:
        """Build a structured RAG prompt"""
        context = "\n\n".join([
            f"[Source {i+1} | Score: {c['score']} | Type: {c['metadata']['type']}]\n{c['chunk']}"
            for i, c in enumerate(retrieved_chunks)
        ])

        prompt = f"""You are a professional stock market analyst AI assistant.
Use ONLY the context below to answer the user's question accurately.
If the answer is not in the context, say "I don't have enough data to answer that."

=== STOCK MARKET CONTEXT ===
{context}
============================

User Question: {query}

Instructions:
- Be precise with numbers (prices, percentages, volumes)
- Reference specific dates when relevant
- Highlight key insights clearly
- If comparing stocks, be systematic

Answer:"""
        return prompt

    def query(self, user_question: str, top_k: int = 5) -> dict:
        """Full RAG pipeline: retrieve → augment → generate"""
        print(f"\n{'='*60}")
        print(f"🔍 Query: {user_question}")
        print(f"{'='*60}")

        # Step 1: Retrieve relevant chunks
        retrieved = self.vector_store.retrieve(user_question, top_k=top_k)
        print(f"\n📚 Retrieved {len(retrieved)} chunks:")
        for i, r in enumerate(retrieved):
            print(f"  [{i+1}] Score={r['score']} | {r['chunk'][:80]}...")

        # Step 2: Build augmented prompt
        prompt = self.build_prompt(user_question, retrieved)

        # Step 3: Generate with Gemini
        print("\n🤖 Generating response with Gemini...")
        response = self.model.generate_content(prompt)

        return {
            'question':        user_question,
            'answer':          response.text,
            'retrieved_chunks': retrieved,
            'sources_count':   len(retrieved)
        }
