#region ############## import and environment variables ##############

from dotenv import load_dotenv
import os
from bidi.algorithm import get_display

from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, AzureAIDocumentIntelligenceLoader
from langchain.prompts import PromptTemplate
import torch

load_dotenv()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion



#region ######## load & split pdf ########################

# file_name = "hazot.pdf"
# project = "hazot"
# path = f"Data/{project}"
# file_path = os.path.join(path, file_name)
file_path = "Data\wrongfull-death\wrongFullDethHeb.pdf"
loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.environ.get("AZURE_API_ENDPOINT"), 
        api_key=os.environ.get("AZURE_API_KEY"), 
        file_path=file_path, 
        api_model="prebuilt-layout",
        mode="markdown", #single, page, or markdown. 
)
docs = loader.load()
torch.save(docs,"Output/WrongfullDeath2.pth")
# docs = torch.load("Output\WrongfullDeath.pth")

with open ("Output/WrongfullDeath2.md", "w", encoding="utf-8") as f:
    f.write(docs[0].page_content)

exit()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
torch.save("Output/hazot2_split.pth")
#endregion
