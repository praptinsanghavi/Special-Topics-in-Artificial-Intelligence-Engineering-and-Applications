from ragas.testset import TestsetGenerator
import inspect

try:
    sig = inspect.signature(TestsetGenerator.generate_with_langchain_docs)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Error inspecting signature: {e}")
