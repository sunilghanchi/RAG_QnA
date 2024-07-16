import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Pinecone as PC
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
import tempfile
import os
from pinecone import Pinecone # type: ignore
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pinecone = Pinecone(api_key=PINECONE_API_KEY)
index_name = "newone"
index = pinecone.Index(index_name)


st.title("PDF Chat QnA")
upload_file = st.file_uploader('Choose a file')

temp_file_path = None
if upload_file is not None:
    if temp_file_path is not None and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    temp_file_path = os.path.join(tempfile.gettempdir(), upload_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(upload_file.getvalue())

    loader = PyPDFLoader(temp_file_path)
    text = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(text)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Custom text mapper function
    def text_mapper(doc):
        return doc.page_content

    docsearch = PC.from_texts([text_mapper(t) for t in docs], embeddings, index_name=index_name)

    retriever = docsearch.as_retriever()

    llm = ChatGroq(
        groq_api_key="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",
        model_name='mixtral-8x7b-32768'
    )

    rag_template = """Answer the question based only on the following context but don't mention that you are using this context, just answer the question: {context} Question: {question} """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()

    user_input = st.text_input("Type your question...")
    if user_input:
        response = rag_chain.invoke(user_input)
        st.write(response)