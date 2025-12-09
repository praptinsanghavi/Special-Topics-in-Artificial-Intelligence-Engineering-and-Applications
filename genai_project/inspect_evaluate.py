from ragas import evaluate
import inspect

try:
    sig = inspect.signature(evaluate)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Error inspecting signature: {e}")
