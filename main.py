import os 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("initializing components...")

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small", # Or any OpenRouter supported embedding model
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False # Crucial for non-OpenAI endpoints
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, 
    openai_api_key=os.environ["OPENROUTER_API_KEY"],    
    openai_api_base="https://openrouter.ai/api/v1",
)



vector_store = PineconeVectorStore(index_name=os.environ['INDEX_NAME'] , embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """ Answer the question based on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
    """
    simple retrieval chian without LCEL.
    manully retrieves documents, formats them, and generage a response.
    """
    #step1: retrieve relevant documents
    docs = retriever.invoke(query)
    # step2
    context = format_docs(docs)
    #step3 format the promot
    messages = prompt_template.format_messages(context = context, question = query)
    response = llm.invoke(messages) 
    return response.content

def retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs 
        ) 
        | prompt_template | llm | StrOutputParser()
    )
    return retrieval_chain 

if __name__ == "__main__":
    print("Retrieving context...")

    query = "what is Pinecone in machine learning?"

    # ==============================
    # Opetino 0, Raw invocation without RAG
    # ==============================
    print("\n\nOption 0: Raw invocation without RAG")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(f"Raw result: {result_raw.content}")

    # ==============================
    # Option 1, RAG with retriever wihtou LCEL 
    # ==============================
    print("\n\n", "=" * 50)
    print("Implementation without LCEL...")
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("Anser: ", result_without_lcel)

    # ==============================
    # Option 2, RAG with retriever with LCEL 
    # ==============================
    print("WITH LCEL =====================")
    chain_with_lcel = retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question":query})
    print("\nAnswer:")
    print(result_with_lcel)
