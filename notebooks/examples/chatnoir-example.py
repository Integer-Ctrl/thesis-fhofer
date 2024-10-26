from chatnoir_pyterrier import ChatNoirRetrieve, Feature
from chatnoir_api import Index

chatnoir = ChatNoirRetrieve(staging=True, num_results=5, index=Index.MSMarcoV21, features=Feature.SNIPPET_TEXT)
print(chatnoir.search("python library"))
