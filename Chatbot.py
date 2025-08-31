import streamlit as st
import PyPDF2
import asyncio
import os
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client using updated interface
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Globals
pdf_chunks = []
vectorizer = None
chunk_embeddings = None

mcp = FastMCP("pdf_docs")

def split_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better context preservation."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@mcp.tool()
async def answer_query(query: str) -> str:
    """Answer questions based on PDF content using TF-IDF and OpenAI."""
    if not pdf_chunks:
        return "No PDF loaded. Please upload a PDF document first."
    
    try:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, chunk_embeddings)
        best_idx = sims.argmax()
        context = pdf_chunks[best_idx]
        
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        response = await asyncio.to_thread(lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        ))
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Retrieval QA Chatbot", page_icon="📚", layout="wide")

st.title("📚 Retrieval QA Chatbot")
st.markdown("Upload a PDF document and ask questions about its content!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1000, value=500, step=100)
    overlap = st.slider("Overlap", min_value=10, max_value=100, value=50, step=10)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload a PDF document to analyze")
    
    if uploaded_file:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            full_text = ""
            progress_bar = st.progress(0)
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                progress_bar.progress((i + 1) / len(reader.pages))
            
            if full_text.strip():
                pdf_chunks = split_text(full_text, chunk_size, overlap)
                vectorizer = TfidfVectorizer().fit(pdf_chunks)
                chunk_embeddings = vectorizer.transform(pdf_chunks)
                
                st.success(f"✅ PDF loaded successfully! Processed {len(pdf_chunks)} chunks.")
                st.info(f"📊 Document stats: {len(reader.pages)} pages, {len(full_text)} characters")
            else:
                st.error("❌ No text could be extracted from the PDF.")
                
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")

with col2:
    st.header("❓ Ask Questions")
    
    if not pdf_chunks:
        st.info("👆 Please upload a PDF document first to start asking questions.")
    else:
        user_query = st.text_input("Ask a question about the document:", placeholder="What is the main topic of this document?")
        
        if user_query:
            with st.spinner("🤔 Thinking..."):
                answer = asyncio.run(answer_query(user_query))
            
            st.write("**💡 Answer:**")
            st.write(answer)
            
            # Show relevant context
            if pdf_chunks:
                q_vec = vectorizer.transform([user_query])
                sims = cosine_similarity(q_vec, chunk_embeddings)
                best_idx = sims.argmax()
                
                with st.expander("🔍 View Relevant Context"):
                    st.text(pdf_chunks[best_idx])

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, OpenAI, and FastMCP")
