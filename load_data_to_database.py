from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from  dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#loading environment variables
load_dotenv()

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
    db= Chroma.from_documents(chunk_list,embeddings_model)
    return db



# create a query function
def get_similarity_search(query,db,k=2):
    """Takes in query creates a query embding and returns response"""
    embed_query = OpenAIEmbeddings().embed_query(query)
    similar_doc= db.similarity_search_by_vector(embed_query,k=k)
    return similar_doc


if __name__ == '__main__':
    file_path='data\old_tax_guide.pdf'
    document= get_source_data_pdf(file_path)
    # print(document)
    # print(type(document))

    doc_chunks= create_chunks(document)
    # print(len(doc_chunks))
    # print(doc_chunks[0])
    # print(type(doc_chunks))

    db = create_Vector_db(doc_chunks)
    query = 'what is income tax'

    ans= get_similarity_search(query,db)
    print(ans)
