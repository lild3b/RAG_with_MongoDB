import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_together import ChatTogether
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

def main():

    load_dotenv()

    # get api keys from env
    mongo_api = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    hf_api = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    together_api = os.getenv("TOGETHER_API_KEY")

    DB_NAME = "database_sample"
    COLLECTION_NAME = "new_data"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "new_data_index"

    def get_embedding_function():
        embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api, model_name="sentence-transformers/all-mpnet-base-v2" 
    )
        return embeddings

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        mongo_api,
        DB_NAME + "." + COLLECTION_NAME,
        get_embedding_function(),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    qa_retriever = vector_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 25},
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    model = ChatTogether(
        together_api_key = together_api,
        model = "meta-llama/Llama-3-70b-chat-hf",
    )

    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=qa_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    docs = qa({"query": "Where does nondual belongs on the locations?"})

    print(docs["result"])
    #print(docs["source_documents"])

if __name__ == "__main__":
    main()