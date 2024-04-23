from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, AzureAIDocumentIntelligenceLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone, ServerlessSpec

import time


load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
project = "reasonability"
file_name = "full-verdict-of-supreme-court-regarding-reasonableness-bill-january-1-2024.pdf"
path = f"Data/{project}"
file_path = os.path.join(path, file_name)
index_name = project
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        name=index_name,
        dimension=3072,  
        metric='cosine',
        spec = ServerlessSpec(
            cloud='aws', 
            region=os.environ.get("PINECONE_ENVIRONMENT_REGION")
        )
    )
    # wait for index to be initialized
    time.sleep(1)

loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.environ.get("AZURE_API_ENDPOINT"), 
        api_key=os.environ.get("AZURE_API_KEY"), 
        file_path=file_path, 
        api_model="prebuilt-layout"
)
raw_documents = loader.load()
print("load")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)
print("split")
for i, doc in enumerate(documents):
    # new_url = 'https://arxiv.org/abs/2312.07305'
    doc.metadata.update({"number": i})
    print(doc)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
PineconeLangChain.from_documents(documents, embeddings, index_name=index_name)



