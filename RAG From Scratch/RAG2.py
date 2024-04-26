
#region  ####### Encoding or Tokenazing #########
question = "What kinds of pets do I like?"
document1 = "My favorite pet is a cat."
document2 = "I like to code in Python."

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    encoded = encoding.encode(string)
    num_tokens = len(encoded)
    return num_tokens, encoded

print(num_tokens_from_string(question, "cl100k_base"))

encoding = tiktoken.get_encoding(encoding_name="cl100k_base")
print(encoding.n_vocab)

for i in range(13100, 13200):
    print (i, encoding.decode([i]))
#endregion

#region ####### Embedding #########
from langchain_openai import OpenAIEmbeddings
embd = OpenAIEmbeddings()
question_result = embd.embed_query(question)
document1_result = embd.embed_query(document1)
document2_result = embd.embed_query(document2)
print (len(question_result), len(document1_result), len(document2_result))
# print (question_result)

import numpy as np

# https://en.wikipedia.org/wiki/Cosine_similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity1 = cosine_similarity(question_result, document1_result)
similarity2 = cosine_similarity(question_result, document2_result)
print("Cosine Similarity:", similarity1, similarity2)
#endregion

#region ###### Indexing #########
# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index & save to disk
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# vectorstore = Chroma.from_documents(documents=splits, 
                                    # embedding=OpenAIEmbeddings(),
                                    # persist_directory="./chroma_db")

# load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
#endregion

#region ###### Retriever ########
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Retrievel
docs = retriever.get_relevant_documents("What are Agents?")
print(docs)
#endregion

#region ######## Generation ##########
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
# LLM
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
# Chain
chain = prompt | llm
# Run
response = chain.invoke({"context":docs,"question":"What is Task Decomposition?"})
print(response)

from langchain import hub
prompt_hub_rag = hub.pull("rlm/rag-prompt")
print(prompt_hub_rag)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
#endregion

response = rag_chain.invoke("What is Task Decomposition?")
print(response)
