from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4-turbo")
prompt = ChatPromptTemplate.from_template("tell me a joke aboaut {topic}")
chain = prompt | model | StrOutputParser()

# print(chain.input_schema.schema())
# print(promppt.input_schema.schema())
# print(model.input_schema.schema())

print(chain.invoke("cats"))


