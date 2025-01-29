#importing all the required modules
import streamlit as st
from PyPDF2 import PdfReader
import requests
from io import BytesIO
import os
import requests
import tempfile
from langchain import  PromptTemplate
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.llms import Ollama
from langchain.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain.document_loaders import PlaywrightURLLoader
from langchain_groq import ChatGroq
from langchain.schema import Document
import dotenv

#creating embeddings to use for url and pdf 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sidebar options
st.sidebar.title("Options")
option = st.sidebar.radio(
    "Select an option:",
    ("Default", "PDF", "URL"),
    index=0
)
st.title("MediBoT")

# Default option
def handle_default():
    PINECONE_INDEX_NAME = "main-project"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    os.environ['PINECONE_API_KEY']="pcsk_2v4NVt_PCxDtbj9z3ABZymMMvbQ7mUbsut1adBZVLvVRZdAAAoHhLrfsNGMWosfDMGAg5"
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME,embedding=embeddings)
    docsearch= PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME,embeddings)
    retriever= docsearch.as_retriever(search_type="similarity", search_kwargs={"k":10})
    #generating using chatgroq
    llm = ChatGroq(
    temperature=0, 
    groq_api_key="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",
      model_name="llama3-8b-8192"
      )
    system_prompt= (
    "You are an assistant for question-answer task."
    "Use the given information to answer the questions."
    "If you don't know the answer , say you don't know."
    "Do not generate questions , just answer the provided question"
    "Try to give accurate answers"
   "{context}" 
   )

    prompt = ChatPromptTemplate.from_messages(
    [
    ("system",system_prompt),
    ("human","{input}")

    ]
    )
    
    query=st.chat_input("Enter your query")
   
    if query:
        QAChain= create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, QAChain)

        response =rag_chain.invoke({"input": query})

    # Store the conversation history
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

        st.session_state.conversation_history.append((query, response["answer"]))

    # Display the conversation
        for question, answer in st.session_state.conversation_history:
            st.write(f"**You:** {question}")
            st.write(f"**MediBot:** {answer}")

    
     
# PDF option
def handle_pdf():

    st.write("You selected the PDF option.")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    #Loading pdf data
    if uploaded_file:
        print(uploaded_file)
        st.success(f"PDF uploaded successfully! Processing {uploaded_file.name} file.....")
        with open("uploaded_file.pdf","wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file is not None:
            pdf_reader = PdfReader(uploaded_file)
            extracted_text = ""

            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"

            docs= [Document(page_content=extracted_text)]

            splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=20)
            doc_chuncks = splitter.split_documents(docs)
        
        docsearch = Chroma.from_documents(documents=doc_chuncks,
                                              embedding=embeddings,
                                              collection_name="PDF_database",
                                              persist_directory="./chroma_db_PDF")
        retriever= docsearch.as_retriever(search_type="similarity", search_kwargs={"k":10})
    
        llm = ChatGroq(
        temperature=0, 
        groq_api_key="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",
          model_name="llama3-8b-8192"
          )
        system_prompt= (
        "You are an assistant for question-answer task."
        "Use the given information to answer the questions."
        "If you don't know the answer , say you don't know."
        "Do not generate questions , just answer the provided question"
        "Try to give accurate answers"
       "{context}" 
       )

        prompt = ChatPromptTemplate.from_messages(
        [
        ("system",system_prompt),
        ("human","{input}")

        ]
        )

        query=st.chat_input("Enter your query")
   
        if query:
            QAChain= create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, QAChain)

            response =rag_chain.invoke({"input": query})

        # Store the conversation history
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []

            st.session_state.conversation_history.append((query, response["answer"]))

        # Display the conversation
            for q, a in st.session_state.conversation_history:
                st.write(f"**You:** {q}")
                st.write(f"**MediBot:** {a}")
#URL option               
def handle_url():
    st.write("You selected the URL option.")
    url = st.text_input("Enter a URL:")
    if url:
        st.success("URL provided:{}".format(url))
        #Loading url data 
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
            data= response.text
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch URL: {str(e)}")
            return None
        if not data:
            st.error("Failed to extract data. ")
            return
        data=[Document(page_content=data)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        doc_chunks= text_splitter.split_documents(data)

        if not doc_chunks:
            st.error("No content to index ")
            return
         
        docsearch= Chroma.from_documents(documents= doc_chunks,
                                         embedding= embeddings,
                                         collection_name="URL_database",
                                         persist_directory="./chroma_db_url")
        st.success("Index loaded successfully")
        retriever= docsearch.as_retriever(search_type="similarity", search_kwargs={"k":10})
    

        llm = ChatGroq(
        temperature=0, 
        groq_api_key="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",
          model_name="llama3-8b-8192"
          )
        system_prompt= (
        "You are an assistant for question-answer task."
        "Use the given information to answer the questions."
        "If you don't know the answer , say you don't know."
        "Do not generate questions , just answer the provided question"
        "Try to give accurate answers"
       "{context}" 
       )

        prompt = ChatPromptTemplate.from_messages(
        [
        ("system",system_prompt),
        ("human","{input}")

        ]
        )

        query=st.chat_input("Enter your query")
   
        if query:
            QAChain= create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, QAChain)

            response =rag_chain.invoke({"input": query})

        # Store the conversation history
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []

            st.session_state.conversation_history.append((query, response["answer"]))

        # Display the conversation 
            for q, a in st.session_state.conversation_history:
                st.write(f"**You:** {q}")
                st.write(f"**MediBot:** {a}")
      
# Handling options

if option == "Default":
    handle_default()
elif option == "PDF":
    handle_pdf()
elif option == "URL":
    handle_url()
