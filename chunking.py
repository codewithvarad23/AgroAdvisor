from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


def txt_data_load(path):
    loader = TextLoader(path)
    data = loader.load()
    return data

def text_split(data):
    sample_data = data[0].page_content if data else ""
    text_chunk = CharacterTextSplitter(chunk_size=200)
    metadata = text_chunk.create_documents([sample_data])
    return metadata[0].page_content if metadata else "No content available."

# Load and split the text
data = txt_data_load('text.txt') # Text-file-Loader only 
split_text = text_split(data)
print("Split Text:\n", split_text)
