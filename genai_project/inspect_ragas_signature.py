from ragas.testset import TestsetGenerator
import inspect

try:
    sig = inspect.signature(TestsetGenerator.from_langchain)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Error inspecting signature: {e}")

print("\nDocstring:")
print(TestsetGenerator.from_langchain.__doc__)
