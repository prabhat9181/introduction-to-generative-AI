from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def make_chunks(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def create_embeddings(chunks):
    return model.encode(chunks)
