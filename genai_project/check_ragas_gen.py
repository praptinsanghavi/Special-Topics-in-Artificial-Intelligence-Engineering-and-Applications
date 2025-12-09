try:
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    print("Ragas TestsetGenerator found.")
except ImportError as e:
    print(f"Import Error: {e}")
    try:
        from ragas.testset import TestsetGenerator
        print("Ragas TestsetGenerator found in root testset.")
    except ImportError as e:
        print(f"Import Error 2: {e}")
