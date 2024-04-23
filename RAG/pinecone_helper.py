from pinecone import Pinecone
from dotenv import load_dotenv
import os
from bidi.algorithm import get_display

def get_all_id (index, namespace=None):

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index)
    lst = []
    for ids in index.list(namespace=namespace):
        lst.append(ids)
    return lst

def get_record_by_id_list(index, id_list, namespace=""):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index)
    response = index.fetch(id_list, namespace=namespace)
    return response

def get_vector_by_id (index, id, namespace=""):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index)
    record = index.fetch([id], namespace=namespace)
    return record['vectors'][id].values

def get_text_by_id (index, id, namespace=""):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index)
    record = index.fetch([id], namespace=namespace)
    text = record['vectors'][id].metadata['text']
    return text

def query_by_vector (index, vector, top_K= 3, namespace=""):
    pass

def query_by_id (index, id, top_K= 3, namespace=""):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index)
    records = index.query(
        namespace=namespace,
        id=id,
        top_k=top_K,
        include_values=True,
        include_metadata=True
    )
    return records.matches

def get_indexes_list ():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    return pc.list_indexes().names()

def main ():

    # index = "wrongfull-death"
    # id0 = '374871bd-4c8d-4381-9c7b-a6f5d28e5972'
    # id1 = 'bf11cac0-6655-4062-bc9f-dd7c2b8e2207'
    # id2= 'dc9ae910-e354-4820-b5b4-32cffa40f491'
    # id11 = '18e64ee1-dad9-4318-b600-ab9d1b70bb7d'
    # id12 = '213e643f-ead7-47e9-8388-46d87d1f22b4'
    # indecise = get_all_id(index)
    # print(indecise)

    # for id in indecise[0]:
    #     print(get_display(get_text_by_id(index=index, id=id)))





    # ind = indecise[0][0]
    # doc = get_record_by_id_list(index, [ind])
    # print('namespace: ', doc['namespace'])
    # print('usage: ', doc['usage'])
    # vectors = doc['vectors']
    # for id in vectors:
    #     # print(type(vectors[id]))
    #     # print(dir(vectors[id]))
    #     # print(vars(vectors[id]))
    #     print('id: ',vectors[id].id)
    #     print('values: ', vectors[id].values)
    #     print('metadata: ', vectors[id].metadata)
    #     print('metadata.text ', vectors[id].metadata['text'])
    
    # print(doc['vectors'])
    # print('#############################################################')
    # str = get_text_by_id(index,id0)
    # print(get_display(str))
    # print(len(str))
    # print('#############################################################')
    # str = get_text_by_id(index,id1)
    # print(get_display(str))
    # print(len(str))
    # print('#############################################################')
    # str = get_text_by_id(index,id2)
    # print(get_display(str))
    # print(len(str))
    # print('#############################################################')
    # str = get_text_by_id(index,id11)
    # print(get_display(str))
    # print(len(str))
    # print('#############################################################')
    # str = get_text_by_id(index,id12)
    # print(get_display(str))
    # print(len(str))

    # print(get_vector_by_id(index=index, id=id))
    # response = query_by_id(index, id)
    # for key in response.to_dict():
    #     print(key)
    # print(len(response.matches))

    # for record in response:
    #     print(record.id)
    #     # print(record.values)
    #     print(record.score)
    #     print(record.metadata['text'])
    print(get_indexes_list())

main()