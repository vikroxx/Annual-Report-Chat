# Annual Report Chat 
This project implements a Retriever-Augmented Generation (RAG) model to interactively discuss the extensive Annual Report of EY-2023. The report, spanning 92 pages, comprehensively details EY's performance across various indicators for the year 2023.

Access the full EY-2023 report [here](https://assets.ey.com/content/dam/ey-sites/ey-com/en_gl/topics/global-review/2023/ey-value-realized-2023-reporting-progress-on-global-impact-v3.pdf).


## Installation
Make sure Python 3.9 or higher is installed on your system. Install the required Python packages with the following command:

`pip install -r requirements.txt`


## Methodology
The methodology involved several key steps:
- **Extraction**: 
Text, images, and graphs were extracted from the report and converted into an easily readable JSON format, which includes metadata such as page numbers. Images and tables were processed separately using GPT-4V, with descriptions and metadata saved accordingly.

- **Organization**:
 The document was segmented according to the Contents page, organizing the data by sections.
- **Analysis**: 
 Each section's data, including text, images, and graphs, was enhanced with metadata. Descriptions for each section were generated using GPT-4.
- **Vectorization**:
 OpenAI embeddings were used to vectorize the document contents, employing advanced concepts from Llamaindex such as *SentenceSplitter*, *Recursive Retrieval*, *QueryEngineTool*, and *RouterQueryEngine*.


## Models Used
The project utilizes **Llama-3 70b**  hosted on groq and **ChatGPT-4**. Users can choose either model as the default for the Query Engine LLM and to generate responses from retrieved results.

Hosted End point - https://ey-chat.translatetracks.com/docs

Interact with the RAG application [here](https://chat-vikrant.streamlit.app/).

Front-End built using Streamlit.

![RAG Application](https://i.ibb.co/0rdjhXp/streamlit.png)
