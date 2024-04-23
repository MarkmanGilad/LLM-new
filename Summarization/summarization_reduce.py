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

file_name = "lior-hamer.pdf"
project = "hamer-lior"
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


#region ################ reduce_documents_chain ################
reduce_template = """"Below is a list of summaries in hebrew that together constitute a summary of a court ruling. 
{docs}
Based on this list, you are required to summarize the ruling in hebrew.Please provide a detailed and comprehensive summary.
The summary should include the following sections, if available:
- The relevant facts.
- The legal or factual question that was to be decided by the court. 
- The positions of the parties to the discussion.
- A summary of the ruling from each of the judges, with a title that includes the name of the judge.
- A summary of the legal principle decided in the ruling. If there was a majority opinion and a minority opinion, please specify which judges were in the majority and which were in the minority."
Helpful Answer:"""

reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
stuff_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)
# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=stuff_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=stuff_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

#endregion

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

response = reduce_documents_chain.invoke(split_docs)
torch.save(response, "Output/hazot.pth")
print(response['output_text'])
