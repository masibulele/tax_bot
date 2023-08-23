from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from  dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

#loading environment variables
load_dotenv()

#create chroma client

collection_name= 'tax_guide_individual'
persist_directory='db'



#get data from  source documents

def get_source_data_pdf(file):
    """takes a pdf document as data  and returns a document object (text and associated metadata) """
    loader = UnstructuredPDFLoader(file)
    data= loader.load()
    return data
    

#break down document in to smaller text chuncks so that model can work with text
def create_chunks(data_list_object):
    """Split the text up into small, semantically meaningful chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20,)
    document= text_splitter.split_documents(data_list_object)
    return document


# create vector representations of the text chunks 
def create_Vector_db(chunk_list):
    """takes in chunk list and creates a vector database"""
    embeddings_model= OpenAIEmbeddings()
    db= Chroma.from_documents(chunk_list,embeddings_model,collection_name=collection_name,persist_directory=persist_directory)
    db.persist()
    return db



# create a query function
def get_similarity_search(query,db,k=2):
    """Takes in query creates a query embding and returns response"""
    embed_query = get_embed_query(query)
    similar_doc= db.similarity_search_by_vector(embed_query,k=k)
    return similar_doc

# Load database from persist database in disk
def load_db_from_disk():
    """Load database from persist database in disk"""
    embedding_model= OpenAIEmbeddings()
    vectordb= Chroma(persist_directory=persist_directory,embedding_function=embedding_model)
    return vectordb

#get collection to query from client
def get_collection(client):
    embeddings= OpenAIEmbeddings()
    tax_collection= client.get_collection(collection_name,embeddings)
    return tax_collection

def get_embed_query(query):
     embed_query=OpenAIEmbeddings().embed_query(query)
     return embed_query


if __name__ == '__main__':
    # file_path='data\Legal-Pub-Guide-IT01-Guide-on-Income-Tax-and-the-Individual.pdf'
    # document= get_source_data_pdf(file_path)
    # # print(document)
    # # print(type(document))

    # doc_chunks= create_chunks(document)
    # # # print(len(doc_chunks))
    # # # print(doc_chunks[0])
    # # # print(type(doc_chunks))

    # db = create_Vector_db(doc_chunks)
    # db=None
    

   

    db= load_db_from_disk()
    qa2 = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever=db.as_retriever())
    query = 'what is income tax'
    response= qa2.run(query)
    print(response)


