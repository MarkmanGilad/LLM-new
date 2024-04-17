from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

#region ####### RTL CSS ##############
st.markdown("""
<style>
body, html {
    direction: RTL;
    text-align: right;
}
p, div, input, label, h1, h2, h3, h4, h5, h6 {
    direction: RTL;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)
#endregion


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


st.title("Chat RAG")

# Initialize chat history
if "history" not in st.session_state:
    history = []
    st.session_state.history = history
else:
    history = st.session_state.history

# Display chat messages from history on app retun
for message in history:
    with st.chat_message(message.type):
        st.markdown(message.content)


######### select pinecone index ###########
def update_session():
    st.session_state["index"]=st.session_state.selected_box
    st.session_state.history = []

indexes = pc.list_indexes().names()

selected_folder = st.sidebar.selectbox('Select a project:', indexes, index=None, key='selected_box', on_change=update_session)

if 'index' in st.session_state:
    
    index_name = st.session_state["index"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    if the answer is not in the following pieces of retrieved context say that the document doesn't has the answer for this question.\
    keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # React to user input
    if question := st.chat_input("Enter your question"):
            
        # Display user message in chat message container
        with st.chat_message("human"):
            st.markdown(question)

        # Display assistant response in chat message container
        
        with st.chat_message("ai"):
            stream = None
            with st.spinner(text="מחפש תשובה במסמך ..."):
                stream = rag_chain.invoke(
                    {"input": question, 
                    "chat_history": history}
                    )
                st.write(stream["answer"])
            

        # Add to chat history
        history.extend([HumanMessage(content=question), AIMessage(content=stream["answer"])])
        st.session_state.history = history