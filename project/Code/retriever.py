import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings):
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding, k)
        return indices[0]
