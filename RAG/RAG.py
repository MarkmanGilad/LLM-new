from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from bidi.algorithm import get_display
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "wrongfull-death"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# retrieved_docs = retriever.invoke("מהי עבירת ההריגה?")
# print(len(retrieved_docs))
# print(get_display(retrieved_docs[0].page_content))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use ten sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='gpt-4')
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
response = rag_chain.invoke("מהו היסוד הנפשי הנדרש לעבירה של המתה בקלות דעת ?")
print(get_display(response))