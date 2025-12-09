try:
    import ragas
    from evaluation import RagasEvaluator
    print("Ragas imported successfully!")
except Exception as e:
    print(f"Ragas import failed: {e}")
