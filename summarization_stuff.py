from dotenv import load_dotenv
import os
from bidi.algorithm import get_display
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, LLMChain, StuffDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, AzureAIDocumentIntelligenceLoader
from langchain.prompts import PromptTemplate
import torch

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")

llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='gpt-4-turbo')

file_name = "hazot.pdf"
project = "hazot"
path = f"Data/{project}"
file_path = os.path.join(path, file_name)

# loader = AzureAIDocumentIntelligenceLoader(
#         api_endpoint=os.environ.get("AZURE_API_ENDPOINT"), 
#         api_key=os.environ.get("AZURE_API_KEY"), 
#         file_path=file_path, 
#         api_model="prebuilt-layout"
# )
# docs = loader.load()

loader = PyPDFLoader(file_path)
docs = loader.load_and_split()

# stuff
summarize_template = """"Below is a list of documents that together constitute a court ruling. 
{docs}
Based on this list, you are required to summarize the ruling in hebrew.Please provide a detailed and comprehensive summary.
The summary should include the following sections:
- The relevant facts.
- The legal or factual question that was to be decided by the court. 
- The positions of the parties to the discussion.
- A summary of the ruling from each of the judges, with a title that includes the name of the judge.
- A summary of the legal principle decided in the ruling. If there was a majority opinion and a minority opinion, please specify which judges were in the majority and which were in the minority."
Helpful Answer:"""
summarize_prompt = PromptTemplate.from_template(summarize_template)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)


# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
stuff_documents_chain = StuffDocumentsChain(
    llm_chain=summarize_chain, document_variable_name="docs"
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=4000, chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

response = stuff_documents_chain.invoke(split_docs)
torch.save(response, "Output/hazot.pth")

import sys
with open("Output/hazot.txt", "w", encoding='utf-8') as f:
    # print(reduce_documents)
    print(response['output_text'], file=f)
