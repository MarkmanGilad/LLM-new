from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from operator import itemgetter

model = ChatOpenAI(model="gpt-4-turbo")

vectorstore = Chroma.from_texts(
    ["harrison worked at kensho", "bears like to eat honey", "harrison lived at Rehovot"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})
print(retriever.invoke("where did harrison worked?"))

template = """Answer the question based only on the following context:
{context}

Question: {question}
Answer in :{language}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()


# chain = {"question":RunnablePassthrough(), "language":itemgetter("language"), "context": itemgetter("question") | retriever  } | prompt | model | output_parser
chain = (
    {   "question":itemgetter("question"), 
        "language":itemgetter("language"), 
        "context": itemgetter("question") | retriever  
    } 
    | prompt 
    | model 
    | output_parser
)
# print(chain.input_schema.schema())
print(chain.invoke({"question":"where did harrison worked?", "language":"French"}))