# from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# runnable = RunnableParallel(
#     passed=RunnablePassthrough(),
#     modified=lambda x: x["num"] + 1,
# )

# print(runnable.invoke({"num": 1}))

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    texts=["harrison worked at kensho", "The cat like milk"], 
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(retrieval_chain.invoke("where did harrison work?"))