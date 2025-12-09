import ragas.testset
try:
    from ragas.testset import synthesizers
    print("Attributes of ragas.testset.synthesizers:")
    print(dir(synthesizers))
except ImportError:
    print("ragas.testset.synthesizers not found.")
