from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import single_hop, multi_hop
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
import random

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

class SyntheticDataGenerator:
    """
    Generates synthetic test data for evaluating the RAG system.
    
    Technology:
        - Uses the `Ragas` framework's `TestsetGenerator`.
        - Generates "Golden Datasets" (Question + Ground Truth) directly from the document corpus.
        
    Evolution Strategies:
        - Single Hop: Questions answerable from a single chunk.
        - Multi Hop: Questions requiring synthesis of information from multiple chunks.
        
    Attributes:
        generator_llm (LangchainLLMWrapper): LLM used to create questions.
        critic_llm (LangchainLLMWrapper): LLM used to validate questions.
        embeddings (LangchainEmbeddingsWrapper): Embedding model for similarity checks.
    """
    
    def __init__(self):
        # Ragas 0.4.x requires explicit wrappers for LangChain objects
        self.generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        self.critic_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        self.embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Initialize Ragas TestsetGenerator with wrapped models
        self.testset_generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.embeddings
        )

    def generate_test_scenarios(self, documents: list[Document], test_size: int = 3):
        """
        Generates synthetic test scenarios using Ragas TestsetGenerator.
        
        Args:
            documents (list[Document]): The source documents to generate questions from.
            test_size (int): Number of test scenarios to generate.
            
        Returns:
            list[dict]: A list of scenarios, each containing 'scenario' (question) 
                        and 'expected_answer_summary' (ground truth).
                        
        Optimization:
            - Sampling: If the document count > 10, we sample random chunks to reduce 
              generation time and cost for the demo.
        """
        try:
            # Ragas generation can be computationally expensive.
            # Limit input docs to 10 chunks for performance during live demos.
            if len(documents) > 10:
                documents = random.sample(documents, 10)
            
            try:
                # Generate testset using Ragas
                testset = self.testset_generator.generate_with_langchain_docs(
                    documents,
                    testset_size=test_size,
                    raise_exceptions=False
                )
            except Exception as e:
                print(f"Ragas internal generation error: {e}")
                testset = None

            # Fallback if Ragas fails (returns None or empty)
            if testset is None:
                raise ValueError("Ragas failed to generate valid testset (likely API connection or Rate Limit issue).")

            # Convert Ragas dataset to application-friendly format
            scenarios = []
            try:
                df = testset.to_pandas()
                for _, row in df.iterrows():
                    scenarios.append({
                        "scenario": row.get('user_input', row.get('question', '')),
                        "expected_answer_summary": row.get('reference', row.get('ground_truth', ''))
                    })
            except Exception as e:
                 print(f"Error parsing Ragas output: {e}")
                 # Fallback if parsing fails but object exists
                 raise e

            return scenarios

        except Exception as e:
            print(f"Ragas generation failed: {e}")
            # Graceful Fallback: Return manual scenarios if Ragas fails
            # This ensures the demo continues even if OpenAI API flakes out
            return [
                {
                    "scenario": "[Fallback] Describe the safety protocols mentioned in the manual.",
                    "expected_answer_summary": "System generated fallback due to API interruption."
                },
                {
                    "scenario": "[Fallback] Identify the main components illustrated in the document.",
                    "expected_answer_summary": "System generated fallback due to API interruption."
                }
            ]
