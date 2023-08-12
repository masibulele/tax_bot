from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from  dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#loading environment variables
load_dotenv()

#get data from  source documents
loader = UnstructuredPDFLoader('data\Tax-guide.pdf')
data= loader.load()
# print(type(data))

#break down document in to smaller text chuncks so that model can work with text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20,)

document= text_splitter.split_documents(data)
# print(document[0])
# print(type(document[1]))

# create vector representations of the text chunks 
embeddings_model= OpenAIEmbeddings()

db= Chroma.from_documents(document,embeddings_model)

query = 'what are the budget proposals'

embed_query = OpenAIEmbeddings().embed_query(query)
docs= db.similarity_search_by_vector(embed_query)
print(docs[0].page_content)