#region ############# import and init #############
from dotenv import load_dotenv
import os
from langchain_core.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI
import sqlite3
from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, 
    MessagesPlaceholder, HumanMessagePromptTemplate)
from langchain.agents import create_tool_calling_agent, create_structured_chat_agent, create_react_agent
from langchain.agents import AgentExecutor
from langchain import hub
import pprint
from langchain.memory import ChatMessageHistory
# from langchain.pydantic_v1 import BaseModel, Field

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion

#region ######## query tool #######################
def sql_query(dB_uri: str, query: str) -> str:
    with sqlite3.connect(dB_uri) as conn:
        c = conn.cursor()
        c.execute(query)
        conn.commit()   
        response = c.fetchall()
    return str(response)

sql_tool = StructuredTool.from_function(
    func=sql_query,
    name="sql_tool",
    description="Execute sqlite query."
)
# print(sql_tool.name)
# print(sql_tool.description)
# print(sql_tool.args)
# print(sql_tool.invoke({"dB_uri": "Data/Users.db", "query": "SELECT * FROM Users"}))
#endregion

#region ######## schema tool ####################
def sql_schema(dB_uri: str) -> str:
    response = {}
    import pandas as pd
    with sqlite3.connect(dB_uri) as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        for table in tables:
            table_name = table[0]
            cursor = conn.execute("SELECT sql FROM sqlite_master WHERE name=?;", [table_name])
            sql = cursor.fetchone()[0]
            sql = " ".join(sql.split())
            response[table_name] = sql
    return response

schema_tool = StructuredTool.from_function(
    func=sql_schema,
    name="sql_schema",
    description="get the schema of all tables in the database."
)
# response = schema_tool.invoke({"dB_uri": "Data/Users.db"})
# print(response)

#endregion


llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
tools = [sql_tool, schema_tool]

#region #### prompt openai-functions-agent ##########
# prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt = ChatPromptTemplate.from_messages([
#   ("system", "You are a helpful assistant"),
#   ("placeholder", "{chat_history}"),
#   ("human", "{input}"),
#   ("placeholder", "{agent_scratchpad}"),
# ])
#endregion

#region #### Gilad Prompt base on openai-functions-agent ##########
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[], 
            template='''You are a helpful assistant who can search db. 
            before any query check the schema of the database.
            use the schema to find the names of tables and columns.
            the user doesn't know the exect name of the columns and tables
            default data base is: Data/Users.db
            always show thoughts''')), 
    MessagesPlaceholder(
        variable_name='chat_history', 
        optional=True), 
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['input'], 
            template='{input}')), 
    MessagesPlaceholder(
        variable_name='agent_scratchpad')
    ])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", '''You are a helpful assistant who can search db.
#     before any query check the schema of the database.
#     use the schema to find the names of tables and columns.
#     the user doesn't know the exect name of the columns and tables'''), 
#     ("placeholder", "{chat_history}"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}")
# ])
#endregion

#region #### prompt react_agent ########
# prompt = hub.pull("hwchase17/react")
# prompt = ChatPromptTemplate.from_messages([
#     ("system",'''Answer the following questions as best you can. 
#         You have access to the following tools:
#         {tools}
#         Use the following format:
#         Question: the input question you must answer
#         Thought: you should always think about what to do
#         Action: the action to take, should be one of [{tool_names}] in 
#         Action Input: the input to the action
#         Observation: the result of the action
#         ... (this Thought/Action/Action Input/Observation can repeat N times)
#         Thought: I now know the final answer
#         Final Answer: the final answer to the original input question
#         Begin!'''), 
#     ("placeholder", "{chat_history}"),
#     ("human", '''
#      {input}
#      {agent_scratchpad}
#      '''),     
# ])
    
    

#endregion

#region #### prompt structured-chat-agent ############
# prompt = hub.pull("hwchase17/structured-chat-agent")
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate(
#         prompt=PromptTemplate(
#             input_variables=[],
#             template = '''
#                 Respond to the human as helpfully and accurately as possible. You have access to the following tools:
#                 {tools}
#                 Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
#                 Valid "action" values: "Final Answer" or {tool_names}
#                 Provide only ONE action per $JSON_BLOB, as shown:
#                 ```
#                 {{
#                 "action": $TOOL_NAME,
#                 "action_input": $INPUT
#                 }}
#                 ```
#                 Follow this format:
#                 Question: input question to answer
#                 Thought: consider previous and subsequent steps
#                 Action:
#                 ```
#                 $JSON_BLOB
#                 ```
#                 Observation: action result
#                 ...(repeat Thought/Action/Observation N times)
#                 Thought: I know what to respond
#                 Action:
#                 ```
#                 {{
#                 "action": "Final Answer",
#                 "action_input": "Final response to human"
#                 }}

#                 Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. 
#                 Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation".
#             ''')),
#     MessagesPlaceholder(
#         variable_name='chat_history', 
#         optional=True), 
#     HumanMessagePromptTemplate(
#         prompt=PromptTemplate(
#             input_variables=['input'], 
#             template='''
#                 {input}
#                 {agent_scratchpad}
#                 (reminder to respond in a JSON blob no matter what and always show thoughts)
#             ''')), 
#    ])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", 
#      '''
#         Respond to the human as helpfully and accurately as possible. You have access to the following tools:
#         {tools}
#         Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
#         Valid "action" values: "Final Answer" or {tool_names}
#         Provide only ONE action per $JSON_BLOB, as shown:
#         ```
#         {{
#         "action": $TOOL_NAME,
#         "action_input": $INPUT
#         }}
#         ```
#         Follow this format:
#         Question: input question to answer
#         Thought: consider previous and subsequent steps
#         Action:
#         ```
#         $JSON_BLOB
#         ```
#         Observation: action result
#         ... (repeat Thought/Action/Observation N times)
#         Thought: I know what to respond
#         Action:
#         ```
#         {{
#         "action": "Final Answer",
#         "action_input": "Final response to human"
#         }}
#         Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. 
#         Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation")
#      '''
#     ),
#     ("placeholder", "{chat_history}"),
#     ("human", '''{input}
     
#      {agent_scratchpad})
#      (reminder to respond in a JSON blob no matter what)''')
# ])

#endregion

# agent = create_react_agent(llm, tools, prompt) # not working
agent = create_tool_calling_agent(llm, tools, prompt)
# agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

prompt1 = """
in Data/Users.db
add this user:
first name: Moshe 
last name: Levy
username: M123
pass: 1968
email: Moshe@gmail.com
admin: false
id: 44
address: 5 Begin st., Tel Aviv, Israel
"""
prompt2 = '''
in Data/Users.db
הדפס את כל המשתמשים בטבלה הכוללת את הכתובות שלהם, לרבות משתמשים ללא כתובת
'''
# print(agent_executor.invoke({"input": f"in 'Data/Users.db' {prompt1}"})["output"])
# print(agent_executor.invoke({"input": "in Data/Users.db fetch all users with adresses"})["output"])
# print(agent_executor.invoke({"input": f"{prompt1}"})["output"])
history = ChatMessageHistory()
while True:
    txt = input ("Enter your prompt:\n")
    response = agent_executor.invoke({"input": txt, "chat_history":history.messages})
    print(response["output"])
    history.add_user_message(txt)
    history.add_ai_message(response["output"])
    # history.add_ai_message(str(response["intermediate_steps"]))
