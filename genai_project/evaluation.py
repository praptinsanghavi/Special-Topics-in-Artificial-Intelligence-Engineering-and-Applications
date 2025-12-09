from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

class RagasEvaluator:
    """
    Evaluator class for assessing RAG system performance using Ragas metrics.
    
    Metrics:
        - Faithfulness: Checks if the answer is grounded in the context.
        - Answer Relevancy: Checks if the answer addresses the user's question.
        - Context Precision: Checks if relevant chunks are ranked highly.
        - Context Recall: Checks if the retrieved context contains the ground truth.
        
    Attributes:
        llm (LangchainLLMWrapper): Wrapped LLM for evaluation logic (judge).
        embeddings (LangchainEmbeddingsWrapper): Wrapped embedding model for similarity metrics.
    """
    
    def __init__(self):
        # Ragas uses OpenAI by default for its "Judge" models.
        # We wrap them to ensure compatibility with Ragas 0.4.x.
        self.llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        self.embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    def evaluate_results(self, questions, answers, contexts, ground_truths=None):
        """
        Runs the Ragas evaluation pipeline on the provided dataset.
        
        Args:
            questions (list): List of user queries.
            answers (list): List of generated answers from the RAG system.
            contexts (list): List of list of retrieved context strings (e.g., [['ctx1', 'ctx2'], ...]).
            ground_truths (list, optional): List of list of ground truth strings. Required for Recall/Precision.
            
        Returns:
            Result: A Ragas Result object containing aggregate scores and per-row metrics.
        """
        
        # Construct the dataset dictionary expected by Ragas
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        # Select metrics based on data availability
        if ground_truths:
            data["ground_truth"] = ground_truths
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        else:
            # Without ground truth, we can only measure faithfulness and relevancy (Reference-Free)
            metrics = [faithfulness, answer_relevancy]

        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_dict(data)
        
        # Execute evaluation
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return results
