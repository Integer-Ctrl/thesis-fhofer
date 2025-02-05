import ir_datasets
from ir_datasets_clueweb22 import register

register()

dataset = ir_datasets.load("clueweb22/b")
docs_store = dataset.docs_store()

docs_dict = docs_store.get_many(['clueweb22-en0024-23-06356', 'clueweb22-en0007-75-03624'])
for docid, doc in docs_dict.items():
    print(doc.default_text())
