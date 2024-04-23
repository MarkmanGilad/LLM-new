from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='gpt-4')
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
parser = StrOutputParser()
chain = prompt | chat | parser

# Initialize chat history
history = ChatMessageHistory()
while True:
    text = input("Enter question: \n")

    history.add_user_message(text)

    response = chain.invoke(
        {
            "messages": history.messages 
        })
    print(response)
            
    history.add_ai_message(response)
    print("#################################################")
    print(history)
    print("#################################################")
