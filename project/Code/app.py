import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="Smart PDF QA", layout="centered")

st.title("🤖 Smart PDF Question Answering System")
st.write("Upload PDF → Ask → Get Smart Answer")

# ✅ load models only once
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, qa_model

embed_model, qa_model = load_models()

# ✅ process PDF only once
@st.cache_data
def process_pdf(file_bytes):
    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)

    reader = PdfReader("temp.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # better chunking
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return chunks, index

# file upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded ✅")

    with st.spinner("Processing PDF..."):
        chunks, index = process_pdf(uploaded_file.read())

    st.success("PDF processed 🚀")

    query = st.text_input("Ask your question:")

    if query:
        # retrieve
        query_embedding = embed_model.encode([query])
        distances, indices = index.search(query_embedding, 5)

        context = " ".join([chunks[i] for i in indices[0]])

        st.subheader("📌 Retrieved Context")
        st.write(context[:400] + "...")

        # ✅ improved prompt
        prompt = f"""
You are an AI assistant.
Give a short and clear answer in 2-3 lines only.

Context:
{context}

Question:
{query}

Answer:
"""

        result = qa_model(prompt, max_length=120, do_sample=False)

        st.subheader("🤖 Final Answer")

        answer = result[0]['generated_text'].strip()

        # clean answer
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        st.write(answer)