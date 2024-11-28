import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_csv_text(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_csv(index=False)

def get_word_text(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = CohereEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Cohere(model="command-xlarge-nightly", temperature=0.5, max_tokens=512)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Your Files")
    st.header("Chat with Your Files 📂")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation.run(user_question)
        st.write(response)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your files (PDF, CSV, DOCX) here and click on 'Process'", 
            accept_multiple_files=True, 
            type=['pdf', 'csv', 'docx']
        )
        if st.button("Process") and uploaded_files:
            with st.spinner("Processing"):
                raw_text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith(".pdf"):
                        raw_text += get_pdf_text([uploaded_file])
                    elif uploaded_file.name.endswith(".csv"):
                        raw_text += get_csv_text(uploaded_file)
                    elif uploaded_file.name.endswith(".docx"):
                        raw_text += get_word_text(uploaded_file)

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
