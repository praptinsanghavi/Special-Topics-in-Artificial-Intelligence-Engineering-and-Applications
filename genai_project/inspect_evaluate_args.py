from ragas import evaluate
import inspect

try:
    sig = inspect.signature(evaluate)
    print("Arguments:")
    for name, param in sig.parameters.items():
        print(f"- {name}")
except Exception as e:
    print(f"Error inspecting signature: {e}")
