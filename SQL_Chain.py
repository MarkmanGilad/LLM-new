#region ############# import and init #############
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from pprint import pprint
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion

db = SQLDatabase.from_uri("sqlite:///Data/Users.db")
#region ############ SQL Queries #######################

# print(db.dialect)
# print(db.get_usable_table_names())
# db.run("CREATE TABLE USERS (first_name text, last_name text)")
# db.run("Insert Into Users Values ('Yoav', 'Cohen')")
# results = db.run("SELECT * FROM Users;",fetch="cursor")
# print(type(results))
# print(results)
# pprint(list(results.mappings()))
# for res in results:
    # print (res)
#endregion

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
chain.get_prompts()[0].pretty_print()

response = chain.invoke({"question": "How many users are there"})
print(response)
print(db.run(response))

#region ######### create query tool and execute ##########

execute_query = QuerySQLDataBaseTool(db=db)
sql_query = create_sql_query_chain(llm, db)
chain = sql_query | execute_query 
response = chain.invoke({"question": "retrieve all users"})
pprint(response)


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=sql_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

response = chain.invoke({"question": "how many users with last name markman? what are their names?"})
print(response)