from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

class DocumentProcessor:
    @staticmethod
    def load_document(file_path):
        """Loads a document based on file extension"""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        return loader.load()

    @staticmethod
    def split_documents(documents):
        """Splits documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=['\n\n', '\n', '(?=>\. )', ' ', '']
        )
        return text_splitter.split_documents(documents)
