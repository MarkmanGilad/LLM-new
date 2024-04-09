from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os


load_dotenv()
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='davinci-002')
# Notice that "chat_history" is present in the prompt template
template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""
prompt = PromptTemplate.from_template(template)
# Notice that we need to align the `memory_key`
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
response = conversation({"question": "explain what is Reinforcement learning"})
print(response['text'])
print(response)
