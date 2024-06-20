import os
from dotenv import load_dotenv
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# documents folder path
DATA_PATH = 'pdf'

def main():
    load_dotenv()

    # initialize MongoDB python client
    client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))

    DB_NAME = "database_sample"
    COLLECTION_NAME = "new_data"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "new_data_index"

    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

    def load_documents():
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()


    def split_documents(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def get_embedding_function():
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model_name="sentence-transformers/all-mpnet-base-v2" 
    )
        return embeddings
    
    # preprocessing docs
    documents = load_documents()
    chunks = split_documents(documents)

    # insert data and embeddings to MongoDB.
    vector_insert = MongoDBAtlasVectorSearch.from_documents(
        documents = chunks,
        embedding = get_embedding_function(),
        collection = MONGODB_COLLECTION,
        index_name = ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    (f"Documents {documents}")
    

if __name__ == "__main__":
    main()