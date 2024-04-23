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

file_name = "hazot.pdf"
project = "hazot"
path = f"Data/{project}"
file_path = os.path.join(path, file_name)

loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.environ.get("AZURE_API_ENDPOINT"), 
        api_key=os.environ.get("AZURE_API_KEY"), 
        file_path=file_path, 
        api_model="prebuilt-layout"
)
# docs = loader.load()
# torch.save(docs,"Output/hazot1.pth")
docs = torch.load("Output/hazot1.pth")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
#endregion

#region ############## prompt ##########################
summarize_template = """"Below is part of a court ruling. 
{text}
Based only on this text, you are required to summarize the text in hebrew. Please provide a detailed and comprehensive summary.
The summary should include the following sections, if available:
- The relevant facts in details.
- All the legal or factual question that was to be decided by the court in details. for every question give a title and a full summary.
- The positions of the parties to the discussion.
- A detailed summary of the ruling from each of the judges, with a title that includes the name of the judge. please provide a very detailed summary.
- A detailed summary of the legal principle decided in the ruling. If there was a majority opinion and a minority opinion, please specify which judges were in the majority and which were in the minority."
Don't use any sources other that the above.
Helpful Answer:"""
summarize_prompt = PromptTemplate.from_template(summarize_template)


refine_template = """"Your job is to produce a final summary in Hebrew.
We have provided an existing summary up to a certain point: 
{existing_answer}
We have the opportunity to refine the existing summary (only if needed) with some more context below:
{text}
Given the new context, refine the original summary in Hebrew. If the context isn't useful, return the original summary.
The summary should include the following sections, if available:
- The relevant facts in details.
- All the legal or factual question that was to be decided by the court in details. for every question give a title and a full summary.
- The positions of the parties to the discussion.
- A detailed summary of the ruling from each of the judges, with a title that includes the name of the judge. please provide a very detailed summary.
- A detailed summary of the legal principle decided in the ruling. If there was a majority opinion and a minority opinion, please specify which judges were in the majority and which were in the minority."
Don't use any sources other that the above.
Helpful Answer:"""

refine_prompt = PromptTemplate.from_template(refine_template)
#endregion

#region ############### LLM & chain ###########################

llm = ChatOpenAI(temperature=0, model='gpt-4-turbo')
chain = load_summarize_chain(
    llm=llm, 
    chain_type="refine", 
    question_prompt=summarize_prompt, 
    refine_prompt=refine_prompt,
    input_key="input_documents",
    output_key="output_text"
    )
response = chain.invoke({"input_documents": split_docs})
print(get_display(response['output_text']))

#endregion


