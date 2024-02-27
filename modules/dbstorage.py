from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma

modelPath = "distiluse-base-multilingual-cased-v1"
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load Documents (data loader)
loader = CSVLoader(file_path='./data/train_data_q2a5.csv', encoding='utf-8')
data = loader.load()

# Load ChromaDb
db = Chroma.from_documents(data, embeddings, persist_directory="./chroma_db_q2a5")