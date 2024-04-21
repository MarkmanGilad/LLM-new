
from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ChatMessageHistory


load_dotenv()

#region ####### Tavily search engine Tool ##################

search = TavilySearchResults()
# response = search.invoke("what is the weather in SF")
# print(response)

#endregion

#region ####### create FAISS Retriever Tool add a web page to vector store #######################

# loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
# docs = loader.load()
# documents = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200
#     ).split_documents(docs)
# vector = FAISS.from_documents(documents, OpenAIEmbeddings())
# retriever = vector.as_retriever()
# vector.save_local("Data/faiss_vector_store")
vector = FAISS.load_local("Data/faiss_vector_store",OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vector.as_retriever()
# response = retriever.get_relevant_documents("how to upload a dataset")
# print(response[0])

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="langsmith_search",
    description="Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
# print(retriever_tool)
# print(retriever_tool.name)
# print(retriever_tool.description)
# print(retriever_tool.args)
# print(retriever_tool.return_direct)
# response = retriever_tool.run("what is langSmith")
# print(response)
#endregion

#region ######### Create Agent with two tool #############
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Get the prompt to use - you can modify this! https://smith.langchain.com/hub/hwchase17/openai-functions-agent
prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt.messages)

tools = [search, retriever_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
#endregion

#region ########## call agent ##############

# history
history = ChatMessageHistory()

# response = agent_executor.invoke({"input": "hi!"})
# print(response)
# response = agent_executor.invoke({"input": "how can langsmith help with testing?"})
# print(response)
response = agent_executor.invoke({"input": "Who is Orly Markman?"})
history.add_user_message(response['input'])
history.add_ai_message(response['output'])
print(response)
response = agent_executor.invoke({"input": "In what year did she born?", "chat_history": history.messages})
print(response)
history.add_user_message(response['input'])
history.add_ai_message(response['output'])
response = agent_executor.invoke({"input": "please search the web again", "chat_history": history.messages})
print(response)

#endregion