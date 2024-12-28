import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import unidecode
from unidecode import unidecode
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import logging
from database import save_chat_history
from langchain.schema import HumanMessage, AIMessage




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_text = extract_text(pdf)
            if pdf_text:
                text += unidecode(pdf_text)
            else:
                logging.warning("Could not extract text from%s", pdf)
        except Exception as e:
            logging.error("Error reading %s: %s", pdf, e)
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
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore




def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain




def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']


    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


    # Convert chat history to a serializable format
    serializable_chat_history = [
        {"role": "user" if isinstance(message, HumanMessage) else "bot", "content": message.content}
        for message in st.session_state.chat_history
    ]


    return serializable_chat_history




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat avec PDFs", page_icon="📄")
    st.write(css, unsafe_allow_html=True)




    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_question" not in st.session_state:
        st.session_state.user_question = ""




    st.header("Agent Conversationnel sur les CVs")




    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


    user_question = st.text_input("Posez les questions sur vos documents:",value=st.session_state.user_question)
    if user_question and not st.session_state.get("processing", False):
        st.session_state.processing = True
        # st.session_state.last_question = user_question
        serializable_chat_history = handle_userinput(user_question)
        save_chat_history(serializable_chat_history)
        st.session_state.processing = False
        st.session_state.user_question = ""


    with st.sidebar:
        st.subheader("Vos documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Traiter"):
            with st.spinner("Traitement en cours..."):
                raw_text = get_pdf_text(pdf_docs)


                text_chunks = get_text_chunks(raw_text)


                vectorstore = get_vectorstore(text_chunks)


                st.session_state.conversation = get_conversation_chain(vectorstore)


            st.success("Traitement fini")


if __name__ == '__main__':
    main()