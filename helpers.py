import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import base64
import requests
from tqdm import tqdm
load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

def directory_from_doc_split(file_path : str):
    assert os.path.exists(file_path)

    with open(file_path, 'r') as fp:
        content = fp.readlines()

    striped_content = [x.strip() for x in content]
    content = [x for x in striped_content if len(x) >0]
    
    section_list = []
    for item in content:
        start_page , end_page, title = item.split("-")[0].strip(), "-".join(item.split("-")[1:]).strip().split("=>")[0].strip(), "-".join(item.split("-")[1:]).strip().split("=>")[1].strip()
        section_list.append( {
            "start" : start_page,
            "end" : end_page,
            "title" : title,
        })
    
    section_list = sorted(section_list, key= lambda x : int(x['start']))
    for x, section in enumerate(section_list):
        section['number'] = x+1

    # print(json.dumps(section_list, indent=2))
    if not os.path.exists("data"):
        os.mkdir("data")

    for section in section_list:
        if not os.path.exists(os.path.join("data" , "section_" + str(section["number"]))):
            os.mkdir(os.path.join("data" , "section_" + str(section["number"])))

    return section_list


def describe_image(image_path : str):

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Describe the image in detail, if it's a table explain the table"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 2000
    }

    print('Processing image {}....'.format(image_path))
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    description = (response.json()["choices"][0]['message']['content'])
    print(description)

    return description


def generate_description_for_images(image_dir):
    for image in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir ,image)
        description = describe_image(image_path)

        assert description is not None

        with open(os.path.join(image_dir , "{}.json".format(image.split(".")[0])), "w") as fp:
            json.dump({"page_number" : int(image.split("_")[0]), "description" : description}, fp, indent=2, ensure_ascii=False)



def section_wise_data_generator(image_dir, file_content_path, data_dir, section_list):
    # Each section has a start and end page number, every json in the image dir has a page number, every content in the file_content has a page number, now we need a data.json for each section, that has all the image data as well as the text from the file_content. every entry in the data.json should have a page number, text, and image_path for the image.

    with open(file_content_path, "r") as fp:
        file_content = json.load(fp)

    for section in tqdm(section_list):
        section_number = section["number"]
        start_page = int(section["start"])
        end_page = int(section["end"])
        section_title = section["title"]

        section_data = {
            "title" : section_title,
            "data" : []
        }

        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir ,image)
            page_number = int(image.split("_")[0])
            if page_number >= start_page and page_number <= end_page:
                with open(os.path.join(image_dir , "{}.json".format(image.split(".")[0])), "r") as fp:
                    image_data = json.load(fp)
                    section_data["data"].append({
                        "page_number" : page_number,
                        "text" : image_data["description"],
                        "image_path" : image_path
                    })

        for content in file_content:
            page_number = int(content["page_number"])
            if page_number >= start_page and page_number <= end_page:
                section_data["data"].append({
                    "page_number" : page_number,
                    "text" : content["text"]})

        with open(os.path.join(data_dir, "section_{}".format(section_number), "data.json"), "w") as fp:
            json.dump(section_data, fp, indent=2, ensure_ascii=False)


def call_gpt(prompt):
    client = OpenAI(api_key=openai_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        model="gpt-4-turbo-preview",
        response_format={"type": "json_object"}
    )

    return chat_completion.choices[0].message.content


def get_sectionwise_description(data_dir):
    for folder in tqdm(os.listdir(data_dir)):
        print("Processing: ",folder)
        data_path = os.path.join(data_dir,folder,"data.json")
        with open(data_path,'r') as f:
            data = json.load(f)
        
        texts = []
        for item in data['data']:
            text = item['text']
            if len(text)  < 200:
                continue
            texts.append(text)
        
        assert len(texts) > 0

        prompt = "\n".join(texts) + "\n" + "Give me the topics that this document covers. Do not describe the topics. Respond in json with keys as 1,2,3... and values as the topics. Limit to 20 topics."
        response = call_gpt(prompt)
        print(response)
        
        description_path = os.path.join(data_dir,folder,'description.json')
        with open(description_path,'w') as f:
            json.dump(json.loads(response),f,indent=2,ensure_ascii=False)

#get_sectionwise_description("data")

# section_wise_data_generator(image_dir="images" , 
#                             file_content_path= "file_content.json", 
#                             data_dir="data", 
#                             section_list=directory_from_doc_split("docs_split.txt"))

# directory_from_doc_split("docs_split.txt")  
