from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Setup
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embeddings
)

# Create dummy document
docs = [Document(page_content="""
Ragas is a framework for evaluating Retrieval Augmented Generation (RAG) pipelines. 
It helps developers generate synthetic test data to evaluate their applications.
The framework provides metrics like faithfulness, answer relevancy, context precision, and context recall.
By using Ragas, you can ensure your RAG system performs reliably and accurately across various scenarios.
This text needs to be sufficiently long to pass the token count check in the TestsetGenerator.
""")]

print("Generating small testset...")
try:
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=1,
        raise_exceptions=False
    )
    
    df = testset.to_pandas()
    print("\nDataFrame Columns:")
    print(df.columns.tolist())
    print("\nFirst row keys:")
    print(df.iloc[0].keys())
    
except Exception as e:
    print(f"Error: {e}")
