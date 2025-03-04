from dotenv import load_dotenv
import streamlit as st  
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from pypdf import PdfReader

# Set your API key
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def summarize_pdf(pdf):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    
    if not text.strip():
        return "Could not extract any text from the PDF. The file might be scanned or protected."
    
    try:
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Define prompt template
        prompt_template = """You are an AI assistant tasked with summarizing documents.

Below is the text extracted from a PDF document:

{context}

Please provide a concise summary of the main points in 5-8 sentences:
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
        
        # Initialize LLM and chain
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1, max_tokens=1000)
        chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
        
        # Get relevant chunks and generate summary
        docs = vectorstore.similarity_search("Summarize this document", k=4)
        return chain.run(input_documents=docs, question="Summarize this document")
        
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="PDF Summarizer")
st.title("PDF Summarizer")
st.write("Upload a PDF document and get an AI-generated summary using Claude.")
st.divider()

pdf = st.file_uploader("Upload your PDF", type="pdf")
if st.button("Generate Summary"):
    if pdf:
        with st.spinner("Analyzing PDF and generating summary..."):
            summary = summarize_pdf(pdf)
            st.subheader("PDF Summary")
            st.write(summary)
    else:
        st.error("Please upload a PDF file!")