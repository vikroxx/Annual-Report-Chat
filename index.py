from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.schema import TextNode
import os
import json
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
import shutil
from tqdm import tqdm

load_dotenv()

def create_documents_from_text(text):
    docs = [Document(text=text)]
    return docs

def add_metadata_to_document(documents,page_number,image_path):
    documents[0].metadata['page_number'] = page_number
    if image_path is not None:
        documents[0].metadata['image_path'] = image_path
    return documents

def create_base_nodes(documents,chunk_size,chunk_overlap):
    node_parser = SentenceSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    base_nodes = node_parser.get_nodes_from_documents(documents)
    for idx,node in enumerate(base_nodes):
        node.id_ = "node-{}".format(str(idx))
    return base_nodes


def create_smaller_index_nodes(base_nodes,chunk_sizes):
    parsers = [SentenceSplitter(chunk_size = c,chunk_overlap=0) for c in chunk_sizes]

    all_nodes = []
    for base_node in base_nodes:
        for parser in parsers:
            sub_nodes = parser.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn,index_id=base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        original_node = IndexNode.from_text_node(base_node,index_id = base_node.node_id)
        all_nodes.append(original_node)

    return all_nodes



def create_index_for_section(section_path, overwrite_index=False):
    index_path = os.path.join(section_path,'index')
    if overwrite_index:
        if os.path.exists(index_path):
            print("Index already exists, deleting and creating new index...")
            shutil.rmtree(index_path)

    if not os.path.exists(index_path):
        os.makedirs(index_path)
    else:
        print("Index already exists, loading from cache...")
        return
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    with open(os.path.join(section_path,'data.json'),'r') as f:
        data = json.load(f)    
    
    print("Creating index for section: ",section_path)
    documents = []
    for item in (data['data']):
        page_number = item['page_number']
        text = item['text']
        image_path = item.get('image_path',None)
        document = create_documents_from_text(text)
        document = add_metadata_to_document(document,page_number,image_path)
        documents.extend(document)
    base_nodes = create_base_nodes(documents,chunk_size=1024,chunk_overlap=0)
    all_nodes = create_smaller_index_nodes(base_nodes,chunk_sizes=[256,512])
    index = VectorStoreIndex(all_nodes,embed_model=embed_model)
    index.storage_context.persist(index_path)
    print("Index created for section: ",section_path)


#if __name__== "__main__":
#    sections = os.listdir("data")
#    for section in tqdm(sections):
#        create_index_for_section(os.path.join('data',section),overwrite_index=True)

# create_index_for_section('data/section_1', overwrite_index=True)