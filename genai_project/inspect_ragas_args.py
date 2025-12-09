from ragas.testset import TestsetGenerator
import inspect

try:
    sig = inspect.signature(TestsetGenerator.generate_with_langchain_docs)
    print("Arguments:")
    for name, param in sig.parameters.items():
        print(f"- {name}")
except Exception as e:
    print(f"Error inspecting signature: {e}")
