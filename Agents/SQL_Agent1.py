#region ############# import and init #############
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion
llm = ChatOpenAI(model="gpt-4-turbo",temperature=0)
db = SQLDatabase.from_uri("sqlite:///Data/Users.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
context = toolkit.get_context() # schema
tools = toolkit.get_tools()
# print(tools)

messages = [
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=SQL_FUNCTIONS_SUFFIX),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    MessagesPlaceholder(variable_name="chat_history"),
]

prompt = ChatPromptTemplate.from_messages(messages)
prompt = prompt.partial(**context)


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
memory = ChatMessageHistory(session_id="test-memory")
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    txt = input ("Enter your prompt:\n")
    response = agent_with_chat_history.invoke({"input": txt}, config={"configurable": {"session_id": "<foo>"}}, )
    print(response["output"])
    memory.add_user_message(txt)
    memory.add_ai_message(response["output"])

