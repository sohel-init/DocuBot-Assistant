from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from config import Config

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        # self.chain = self._initialize_chain_with_hf()
        self.chain = self._initialize_chain_with_groq()

    # HF model
    def _initialize_chain_with_hf(self):
        """Initializes the LLM using HF and QA chain"""
        llm = HuggingFaceEndpoint(
            # repo_id=Config.REPO_ID,
            endpoint_url=f"{Config.HF_ENDPOINT_URL}/{Config.REPO_ID}",
            temperature=Config.TEMPERATURE,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            return_full_text=Config.RETURN_FULL_TEXT,
            model_kwargs=Config.MODEL_KWARGS,
            huggingfacehub_api_token=Config.HF_TOKEN
        )
        return load_qa_chain(llm, chain_type="stuff")

    # Groq model
    def _initialize_chain_with_groq(self):
        """Initializes the LLM using Groq and QA chain"""
        llm = ChatGroq(
            temperature=Config.TEMPERATURE,
            model_name=Config.MODEL_NAME,
            api_key=Config.GROQ_API_KEY
        )
        
        return load_qa_chain(llm, chain_type="stuff")

    def create_vector_store(self, docs):
        """Creates a FAISS vector store from split documents"""
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        return self.vector_store

    def query(self, question):
        """Performs similarity search and invokes the chain"""
        if not self.vector_store:
            return "Please upload and process a document first"
        
        # Retriever - Find relevant chunks
        docs = self.vector_store.similarity_search(question)
        
        # Generator - Send chunks & question to LLM
        response = self.chain.invoke(
            input={"input_documents": docs, "question": question}
        )
        return response['output_text']
