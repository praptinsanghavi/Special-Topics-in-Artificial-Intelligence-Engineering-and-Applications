import ragas.testset


print("Attributes of ragas.testset:")
print(dir(ragas.testset))

try:
    from ragas.testset import evolutions
    print("\nragas.testset.evolutions found.")
    print(dir(evolutions))
except ImportError:
    print("\nragas.testset.evolutions NOT found.")

# Try to find where simple, reasoning, multi_context might be
print("\nSearching for 'simple', 'reasoning', 'multi_context'...")
import pkgutil
import ragas

def find_module_with_attr(package, attr_name):
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = __import__(modname, fromlist=[attr_name])
            if hasattr(module, attr_name):
                print(f"Found '{attr_name}' in {modname}")
        except Exception:
            pass

find_module_with_attr(ragas, "simple")
find_module_with_attr(ragas, "reasoning")
find_module_with_attr(ragas, "multi_context")
