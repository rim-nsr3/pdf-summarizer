import streamlit as st 
import os 

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_anthropic import ChatAnthropic
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

def process_text(text): 
  text_splitter = CharacterTextSplitter( 
    separator="\n", 
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len 
  )

  chunks = text_splitter.split_text(text)
  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 

  knowledgeBase = FAISS.from_texts(chunks, embeddings) 
  return knowledgeBase 

def summarizer(pdf): 
    pdf_reader = PdfReader(pdf) 
    text = "" 

  # Extract text from each page of the PDF 
    for page in pdf_reader.pages: 
        text += page.extract_text() or "" 

    knowledgeBase = process_text(text) 
    query = "Summarize the content of the uploaded PDF file in approximately 5-8 sentences." 

    # Load the question and answer chain
    docs = knowledgeBase.similarity_search(query) 
    
    # Initialize Anthropic model
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
    
    # Load the question and answer chain
    chain = load_qa_chain(llm, chain_type='stuff') 

    # Run the chain with the retrieved documents
    response = chain.run(input_documents=docs, question=query)
    return response 

def main():
    st.set_page_config(page_title="PDF Summarizer") 

    st.title("PDF Summarizer") 
    st.write("Summarize your PDF files using Claude by Anthropic!") 
    st.divider() 
    
    pdf = st.file_uploader("Upload your PDF", type="pdf")  
    submit = st.button("Generate Summary") 
    
    if submit:
        if pdf is not None: 
            with st.spinner("Generating summary..."):
                response = summarizer(pdf) 
                st.subheader("PDF Summary")
                st.write(response) 
        else: 
            st.error("Please upload a PDF file!")

if __name__ == "__main__":
    main()