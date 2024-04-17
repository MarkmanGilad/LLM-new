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
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API_KEY")
#endregion

#region ######## load & split pdf ########################
file_name = "wrongFullDethHeb.pdf"
project = "wrongfull-death"
path = f"Data/{project}"
file_path = os.path.join(path, file_name)

loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.environ.get("AZURE_API_ENDPOINT"), 
        api_key=os.environ.get("AZURE_API_KEY"), 
        file_path=file_path, 
        api_model="prebuilt-layout",
        mode="single" # mode="page"
)
# docs = loader.load()
# print("document loaded")
# torch.save(docs, "Output/WrongfullDeath.pth")
docs = torch.load("Output/WrongfullDeath.pth")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=4000, chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)
#endregion

#region ############## prompt ##########################
question_template = """Below is a part of a long document in Hebrew.
{text}
Your task is to read the text and divide it into topics according to the order of the content in the text itself.
If there is a title in the text, it should be considered the start of a new topic.
For each topic discussed in the text, you must specify in hebrew:
- The title of the topic
- A brief summary of the topic.
Helpful Answer:"""
question_prompt = PromptTemplate.from_template(question_template)


refine_template = """"Your job is to produce a final integrated list of topics according to the order of the content in a long document in hebrew.
If there is a title in the text, it should be considered the start of a new topic.
For each topic discussed in the text, you must specify in hebrew:
- The title of the topic
- A brief summary of the topic.
We have provided an existing list of topics up to a certain point in the long document: 
{existing_answer}
We have the opportunity to refine or add topics to the existing list (only if needed) using Using parts from the continuation of the document below:
{text}
Given the new context, divide it into topics according to the order of the content in the text itself, refine or add topics to the existing list in Hebrew. If the context isn't useful, return the original list.
In your answer, detail the entire integrated list of topics from the begining of the document to the end.
Helpful Answer:"""

refine_prompt = PromptTemplate.from_template(refine_template)
#endregion

#region ############### LLM & chain ###########################


llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],temperature=0, model='gpt-4-turbo')
chain = load_summarize_chain(
    llm=llm, 
    chain_type="refine", 
    question_prompt=question_prompt, 
    refine_prompt=refine_prompt,
    return_intermediate_steps= False,
    input_key="input_documents",
    output_key="output_text"
    )
response = chain.invoke({"input_documents": split_docs})
print(get_display(response['output_text']))

#endregion


