
import os
import pandas as pd
from dotenv import load_dotenv
from rag_engine import RAGEngine
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompt_manager import PromptManager

# Load env variables (API Keys)
load_dotenv()

def create_dummy_manual():
    """Creates a simple dummy manual for testing."""
    content = """
    Technical Manual: Industrial Hydraulic Pump X-200
    
    Section 1: Installation
    To install the X-200, first ensure the base is level. Bolt the unit using 4x M10 bolts.
    Connect the inlet hose to port A and the outlet to port B.
    Torque all fittings to 45 Nm.
    
    Section 2: Operation
    Turn the main switch to ON. The green LED indicates power.
    If the red LED flashes, check the oil level.
    The operating pressure range is 20-50 bar.
    
    Section 3: Troubleshooting
    Error E1: Low Voltage. Check the power supply unit.
    Error E2: High Temperature. Clean the cooling fins.
    Noise: Grinding noise indicates bearing failure. Replace Part #B-199.
    """
    with open("dummy_manual.txt", "w") as f:
        f.write(content)
    return "dummy_manual.txt"

def main():
    print("🚀 Starting Quick Evaluation Run...")
    
    # 1. Setup RAG Engine with Dummy Data
    print("Step 1: Ingesting Document...")
    manual_path = create_dummy_manual()
    rag = RAGEngine()
    docs = rag.load_documents(manual_path)
    rag.process_documents(docs)
    print("✓ Document Indexed.")

    # 2. Define Test Dataset (Questions + Ground Truths)
    test_questions = [
        "What is the operating pressure range?",
        "How do I fix Error E2?",
        "What does a grinding noise mean?",
        "What is the torque specification for fittings?"
    ]
    
    ground_truths = [
        "The operating pressure range is 20-50 bar.",
        "To fix Error E2 (High Temperature), clean the cooling fins.",
        "Grinding noise indicates bearing failure, requiring replacement of Part #B-199.",
        "Torque all fittings to 45 Nm."
    ]
    
    # 3. Generate Answers
    print("Step 2: Generating Answers...")
    answers = []
    contexts = []
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt_template = PromptManager.get_technical_support_prompt()
    chain = prompt_template | llm | StrOutputParser()

    for q in test_questions:
        # Retrieve
        retrieved_docs = rag.retrieve(q)
        ctx = [d.page_content for d in retrieved_docs]
        contexts.append(ctx)
        
        # Generate
        context_str = "\n".join(ctx)
        ans = chain.invoke({"context": context_str, "input": q, "chat_history": []})
        answers.append(ans)
        print(f"  Q: {q}\n  A: {ans[:50]}...")

    # 4. Run Ragas Evaluation
    print("Step 3: Evaluating Performance...")
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    # Using default metrics for quick run
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    
    print("\n📊 Evaluation Results:")
    print(results)
    
    # Save results
    results.to_pandas().to_csv("evaluation_results.csv", index=False)
    print("✓ Results saved to 'evaluation_results.csv'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Evaluation Failed: {e}")
