#region ############## import and environment variables ##############

from dotenv import load_dotenv
import os
from bidi.algorithm import get_display

from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import torch

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion

#region ######## load & split pdf ########################

file_name = "lifshits1.pdf"
project = "lifshits"
path = f"Data/{project}"
file_path = os.path.join(path, file_name)

loader = PyPDFLoader(file_path)
docs = loader.load_and_split()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)
#endregion

#region ##############prompt ##########################
summarize_template = """"Below is a list of documents that together constitute a court ruling. 
{text}
Based on this list, you are required to summarize the ruling in hebrew.Please provide a detailed and comprehensive summary.
The summary should include the following sections:
- The relevant facts.
- The legal or factual question that was to be decided by the court. 
- The positions of the parties to the discussion.
- A summary of the ruling from each of the judges, with a title that includes the name of the judge.
- A summary of the legal principle decided in the ruling. If there was a majority opinion and a minority opinion, please specify which judges were in the majority and which were in the minority."
Helpful Answer:"""
summarize_prompt = PromptTemplate.from_template(summarize_template)
#endregion

llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='gpt-4-turbo')
chain = load_summarize_chain(llm, chain_type="stuff", prompt=summarize_prompt)
response = chain.invoke(split_docs)

print(get_display(response['output_text']))