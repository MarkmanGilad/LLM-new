from dotenv import load_dotenv
import os



load_dotenv()

def Tavily_tool ():
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults()
    # response = search.invoke("what is the weather in SF")
    # print(response)
    return search

def Retriever_tool():
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
    vector = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    # response = retriever.get_relevant_documents("how to upload a dataset")
    # print(response[0])

    from langchain.tools.retriever import create_retriever_tool

    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )
    # print(retriever_tool)
    # print(retriever_tool.name)
    # print(retriever_tool.description)
    # print(retriever_tool.args)
    # print(retriever_tool.return_direct)
    # response = retriever_tool.run("what is langSmith")
    # print(response)
    return retriever_tool

def Create_Agent():
    from langchain_openai import ChatOpenAI
    from langchain import hub

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Get the prompt to use - you can modify this! https://smith.langchain.com/hub/hwchase17/openai-functions-agent
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # print(prompt.messages)
    
    from langchain.agents import create_openai_functions_agent
    search = Tavily_tool()
    retriever_tool = Retriever_tool()
    tools = [search, retriever_tool]
    agent = create_openai_functions_agent(llm, tools, prompt)

    from langchain.agents import AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": "hi!"})
    print(response)
    response = agent_executor.invoke({"input": "how can langsmith help with testing?"})
    print(response)
    response = agent_executor.invoke({"input": "whats the weather in sf?"})
    print(response)
# Tavily_tool ()
# retriever_tool()
Create_Agent()
