import os
from llama_index.core import StorageContext, load_index_from_storage
import json
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from fastapi import FastAPI
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

app = FastAPI()


# load_dotenv()

print("Groq_api_key : ",os.environ['GROQ_API_KEY'][0:5])
print("Openai_api_key : " ,os.environ['OPENAI_API_KEY'][0:5])

# Settings.llm = OpenAI(model='gpt-4-turbo-preview', api_key= os.environ['OPENAI_API_KEY'])
Settings.llm = Groq(model="llama3-70b-8192", api_key=os.environ['GROQ_API_KEY'])
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=os.environ['OPENAI_API_KEY'])

def description_json_to_str(description):
    new_description = ""
    for key,value in description.items():
        new_description += "{}\n".format(value)
    return new_description


def reformat_description(section):
    with open(os.path.join('data',section,'data.json'),'r') as f:
        data = json.load(f)    

    with open(os.path.join('data',section,'description.json'),'r') as f:
        description = json.load(f)  

    title = data['title']
    new_description = "Useful for questions related to {}.\n".format(title) 
    new_description += 'Topics covered in this document are:\n{}'
    new_description += description_json_to_str(description)

    return new_description


def create_index_and_description_dict():
    sections = os.listdir("data")
    index_dict = {}
    description_dict = {}
    for section in sections:
        index_dir = os.path.join('data', section,'index')
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        index_dict[section] = index
        description_dict[section] = reformat_description(section)
    return index_dict,description_dict

def init_index():
    index_dict,description_dict = create_index_and_description_dict()
    query_engine_dict = {key:value.as_query_engine() for key,value in index_dict.items()}
    tool_dict = {}
    for key,value in query_engine_dict.items():
        tool_dict[key] = QueryEngineTool.from_defaults(
                    query_engine=query_engine_dict[key],
                description=(
                    description_dict[key]
                ),
            )
    
    query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(llm=OpenAI(model='gpt-4-turbo-preview'),max_outputs=2),
    query_engine_tools=[tool_dict[key] for key in tool_dict.keys()],
    verbose=True
    )
    return query_engine

def get_images_from_source_nodes(source_nodes):
    images = []
    for node in source_nodes:
        if "image_path" in node.metadata.keys():
            if node.metadata["image_path"].split(".")[-1] =="json":
                images.append(".".join(node.metadata["image_path"].split(".")[:-1])+ ".png")
            else:
                images.append(node.metadata["image_path"])
    return list(set(images))

def get_page_numbers_from_source_nodes(source_nodes):
    page_numbers = []
    for node in source_nodes:
        if "page_number" in node.metadata.keys():
            page_numbers.append(str(node.metadata["page_number"]))
    return page_numbers

engine = init_index()

@app.get("/query")
def query_engine(query:str):
    query += "Answer in detail from the context provided. Be extremely detailed and decriptive about your answer."
    try:
        response = engine.query(query)
        images = get_images_from_source_nodes(response.source_nodes)
        page_numbers = get_page_numbers_from_source_nodes(response.source_nodes)
        page_number_string = ""
        if len(page_numbers) > 0:
            page_number_string = "\n\nREFERNCE PAGE NUMBERS: {}".format(", ".join(list(set(page_numbers))))
        
    except ValueError:
        response = "The given question cannot be answered using the provided context."
        images = []
    return {"response":str(response) + "\n" + page_number_string,'images':images}
