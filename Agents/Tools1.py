#region ############# import and init #############
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.invoke({"first_int": 4, "second_int": 5}))

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm_with_tools = llm.bind_tools([multiply])

# print(llm.invoke("Hi").content)
from operator import itemgetter

# chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
# response = chain.invoke("What's four times 23")
# print(response)

response = llm_with_tools.invoke("calc 4 * 12")
print (response)


