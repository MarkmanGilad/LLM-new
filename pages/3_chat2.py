from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

st.title("Gilad ChatGPT")
chat = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='gpt-4')
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
parser = StrOutputParser()
chain = prompt | chat | parser

# Initialize chat history
if "history" not in st.session_state:
    history = ChatMessageHistory()
    st.session_state.history = history
else:
    history = st.session_state.history

# Display chat messages from history on app rerun
for message in history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# React to user input
if prompt := st.chat_input("What is up?"):
    history.add_user_message(prompt)
    
    # Display user message in chat message container
    with st.chat_message("human"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    
    with st.chat_message("ai"):
        stream = chain.stream(
            {
               "messages": history.messages 
            }
        )
        response = st.write_stream(stream)
        

    # Add to chat history
    history.add_ai_message(response)
    st.session_state.history = history