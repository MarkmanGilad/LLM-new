#region ############# import and init #############
from dotenv import load_dotenv
import os

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
tools = [TavilySearchResults(max_results=3)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# print(llm.invoke("Who is Orly Markman?"))

response = agent_executor.invoke({"input": "Who is advocate Gilad Markman ?"})
print(response)

