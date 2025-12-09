from ragas.testset import TestsetGenerator
import inspect

print("TestsetGenerator methods:")
print(dir(TestsetGenerator))

if hasattr(TestsetGenerator, 'generate_with_langchain_docs'):
    print("\nSignature of generate_with_langchain_docs:")
    print(inspect.signature(TestsetGenerator.generate_with_langchain_docs))
else:
    print("\ngenerate_with_langchain_docs NOT found.")
