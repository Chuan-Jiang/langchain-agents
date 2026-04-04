import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    print(os.environ["PINECONE_API_KEY"])

    documents = TextLoader("mediumblog1.txt").load()
    #print(documents)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks") 

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small", # Or any OpenRouter supported embedding model
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False # Crucial for non-OpenAI endpoints
    )
    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, 
        index_name = os.environ['INDEX_NAME'])