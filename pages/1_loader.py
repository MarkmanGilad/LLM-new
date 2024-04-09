from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, AzureAIDocumentIntelligenceLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
import time
from streamlit_pdf_viewer import pdf_viewer

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if 'project' not in st.session_state:
    st.session_state['project'] = ""


#region ############ Choose projects folders #######################
def list_folders(path):
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

def update_session():
    st.session_state["project"]=st.session_state.selected_box

directory_path = "Data"
folders = list_folders(directory_path)
st.sidebar.title('List of Projects')
if folders:
    selected_folder = st.sidebar.selectbox('Select a project:', folders, index=None, key='selected_box', on_change=update_session)

with st.sidebar.popover("New Project"):
    project = st.text_input("Enter project name")
    project_path = f"Data/{project}"
    if st.button('save'):
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        else:
            st.write('project allready exists')
        st.session_state["project"]=project
        st.rerun()    
#endregion
        
st.title(f"Project: {st.session_state['project']}")

#region ############ load new Documnt ###############

def save_uploaded_file(uploaded_file):
    path = f"Data/{st.session_state['project']}"
    if not os.path.exists(path) or st.session_state['project'] == "":
        return None
    
    file_path = os.path.join(path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
if uploaded_file is not None:
    index_name = st.session_state['project']
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
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"File saved at {file_path}")
    
    loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.environ.get("AZURE_API_ENDPOINT"), 
        api_key=os.environ.get("AZURE_API_KEY"), 
        file_path=file_path, 
        api_model="prebuilt-layout"
)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        # new_url = 'https://arxiv.org/abs/2312.07305'
        # doc.metadata.update({"source": new_url})
        pass
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    PineconeLangChain.from_documents(documents, embeddings, index_name=index_name)
    st.success(f"File was embedded in {index_name}")
    st.title("pdfviewer")
    binary_data = uploaded_file.getvalue()
    pdf_viewer(input=binary_data, width=700)
    st.title("iframe")
    st.markdown(f'<iframe src="{file_path}" width="700" height="1000"></iframe>', unsafe_allow_html=True)

#endregion