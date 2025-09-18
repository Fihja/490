import streamlit as st
import ollama
import chromadb
import pdfplumber

st.title("Government Paperwork RAG Chatbot")

# load pdf into chunks
def load_pdf_chunks(pdf_path, chunk_size=500, overlap=50):
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i+chunk_size])
                    text_chunks.append(chunk)
            else:
                st.write(f"⚠️ Page {page_num} had no extractable text.")
    return text_chunks
# file
pdf_path = r"C:\Users\fihja\Downloads\i1040gi.pdf"
docs = load_pdf_chunks(pdf_path)
st.write(f"✅ Loaded {len(docs)} chunks from PDF.")

# Initialize Chromadb and create the collection
client = chromadb.PersistentClient(path="./mydb/")
collection = client.get_or_create_collection(name="gov_docs")

# Precompute document embeddings and add to the collection
st.write("⚙️ Generating embeddings, please wait...")
for i, chunk in enumerate(docs[:]):
    response = ollama.embed(model="nomic-embed-text", input=chunk)
    embeddings = response["embeddings"]
    collection.add(
        ids=[str(i)],
        embeddings=embeddings,
        documents=[chunk]
    )

# Function to handle context retrieval and response generation
def get_relevant_context(prompt, n_results=2):
    prompt_response = ollama.embed(model="nomic-embed-text", input=prompt)
    prompt_embedding = prompt_response["embeddings"]
    results = collection.query(
        query_embeddings=prompt_embedding,
        n_results=n_results
    )
    if results and "documents" in results:
        return [doc for sublist in results["documents"] for doc in sublist]
    return []

# Streamlit UI
user_prompt = st.text_area("Enter a prompt to retrieve context:", height=200)

if st.button("Retrieve Context"):
    contexts = get_relevant_context(user_prompt)
    if contexts:
        st.subheader("Relevant Context(s):")
        for c in contexts:
            st.write(c)

        st.subheader("Generated Response:")
        context_text = " ".join(contexts)
        response = ollama.generate(
            model="llama2",
            prompt=f"Using this data: {context_text}. Respond to this prompt: {user_prompt}"
        )
        st.write(response.get('response', 'No response generated'))
    else:
        st.write("⚠️ No relevant context found.")
