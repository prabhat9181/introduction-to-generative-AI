from pdf_loader import read_pdf
from embedder import make_chunks, create_embeddings
from retriever import VectorStore
from qa_system import generate_answer
from sentence_transformers import SentenceTransformer

text = read_pdf("sample.pdf")
chunks = make_chunks(text)
embeddings = create_embeddings(chunks)

db = VectorStore(embeddings)

query = input("Ask your question: ")

model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode([query])

indices = db.search(query_embedding)
context = " ".join([chunks[i] for i in indices])

answer = generate_answer(context, query)

print("\nAnswer:\n", answer)
