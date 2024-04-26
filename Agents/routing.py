#region ############# import and init #############
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
#endregion


chain = (PromptTemplate.from_template(
    """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.

    Do not respond with more than one word.

    <question>
    {question}
    </question>

    Classification:"""
    )
    | ChatAnthropic(model_name="claude-3-haiku-20240307")
    | StrOutputParser())

print(chain.invoke({"question": "how do I call Anthropic?"}))

langchain_chain = (PromptTemplate.from_template(
    """You are an expert in langchain. 
    Always answer questions starting with "As Harrison Chase told me". 
    Respond to the following question:

    Question: {question}
    Answer:"""
    ) 
    | ChatAnthropic(model_name="claude-3-haiku-20240307"))

anthropic_chain = (PromptTemplate.from_template(
    """You are an expert in anthropic. \
    Always answer questions starting with "As Dario Amodei told me". \
    Respond to the following question:

    Question: {question}
    Answer:"""
    ) 
    | ChatAnthropic(model_name="claude-3-haiku-20240307"))

general_chain = (PromptTemplate.from_template(
    """Respond to the following question:

    Question: {question}
    Answer:"""
    ) 
    | ChatAnthropic(model_name="claude-3-haiku-20240307"))

def route(info):
    if "anthropic" in info["topic"].lower():
        return anthropic_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain
runnablRoute = RunnableLambda(route)

full_chain = (
    {"topic": chain, "question": RunnablePassthrough()} 
    | runnablRoute
    )

full_chain = (
    {"topic": chain, "question":  lambda x: x["question"]} 
    | runnablRoute
    )


print(full_chain.invoke({"question": "how do I use Anthropic?"}))
print(full_chain.invoke({"question": "how do I use LangChain?"}))
print(full_chain.invoke({"question": "whats 2 + 2"}))