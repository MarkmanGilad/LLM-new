from dotenv import load_dotenv
import os
from bidi.algorithm import get_display
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, LLMChain, StuffDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, AzureAIDocumentIntelligenceLoader
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")

llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='gpt-4-turbo')

file_name = "hazot.pdf"
project = "Hazot"
path = f"Data/{project}"
file_path = os.path.join(path, file_name)

# loader = AzureAIDocumentIntelligenceLoader(
#         api_endpoint=os.environ.get("AZURE_API_ENDPOINT"), 
#         api_key=os.environ.get("AZURE_API_KEY"), 
#         file_path=file_path, 
#         api_model="prebuilt-layout",
#         mode="single" # mode="page"
# )
# docs = loader.load()

loader = PyPDFLoader(file_path)
docs = loader.load_and_split()

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes in hebrew
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes in hebrew. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)


summarize = map_reduce_chain.invoke(split_docs)
print(summarize)
print(get_display(summarize['output_text']))

