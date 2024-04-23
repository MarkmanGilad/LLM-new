from dotenv import load_dotenv
import os

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory

load_dotenv()

# memory
history = ChatMessageHistory()
history.add_user_message("what is LangChain?")
history.add_ai_message("LangChain is a platform for building applications with LLMs (Large Language Models) through composability. It allows for the creation of applications that utilize LLMs for tasks such as retrieval augmented generation, analyzing structured data, and building chatbots. LangChain is an open-source project with extensive documentation and is open to contributions from the community. You can find more information about LangChain on their GitHub page: [LangChain GitHub](https://github.com/langchain-ai/langchain)'}")


# Initialize Tools
tools = [TavilySearchResults(max_results=1)]
# Get the prompt to use - you can modify this!

# Create Agent
prompt = hub.pull("hwchase17/openai-tools-agent")
# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Run Agent
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke(
    {
    "input": "what can I do with it?",
    "chat_history": history.messages
    }
    )
print(response)
print(response["output"])
