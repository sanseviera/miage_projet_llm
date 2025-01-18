from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.schema import Document
import os
import shutil
import logging

class RAGService:
    def __init__(self, persist_dir: str = "./data/vectorstore"):
        """
        Initialise le service RAG avec un vector store persistant
        
        Args:
            persist_dir: Chemin où persister le vector store
        """
        self.persist_dir = persist_dir
        
        # Création du dossier de persistance s'il n'existe pas
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Chargement d'un vector store existant ou création d'un nouveau
        if os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3")):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = None

    async def load_and_index_texts(self, texts: List[str], clear_existing: bool = False) -> None:
        """
        Charge et indexe une liste de textes
        
        Args:
            texts: Liste de textes à indexer
            clear_existing: Si True, supprime l'index existant avant d'indexer
        """
        # Nettoyage optionnel de l'existant
        if clear_existing and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir)
            self.vector_store = None
            
        # Découpage des textes
        splits = self.text_splitter.split_text("\n\n".join(texts))
        
        if self.vector_store is None:
            # Création initiale
            indexed_texts = [{"content": text, "timestamp": i} for i, text in enumerate(splits)]
            self.vector_store = Chroma.from_texts(
                indexed_texts,
                splits,
                self.embeddings,
                persist_directory=self.persist_dir
            )
            logging.info("Documents indexed: %s", splits)
        else:
            # Ajout à l'existant
            self.vector_store.add_texts(splits)
            logging.info(f"Additional documents indexed: {splits}")
            
        # Persistance explicite
        self.vector_store.persist()
    
    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Effectue une recherche par similarité
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            
        Returns:
            Liste des documents les plus pertinents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please add documents first.")
            
        results = self.vector_store.similarity_search(query, k=k)
        logging.info(f"Similarity search results: {[doc.page_content for doc in results]}")
        # return [doc.page_content for doc in results]
        return results
    
    def get_all_documents(self) -> List[str]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please add documents first.")
        
        # return [doc.page_content for doc in self.vector_store.similarity_search("", k = 100)]
        # results = self.vector_store.similarity_search("", k=100)
        # documents = [doc.page_content for doc in results]
        # for i in range(len(documents)):
        #     documents[i] = documents[i].replace("\n", " ")
        # print("Documents:", documents)
        # return documents
        documents = []
        # Recherche de tous les documents en utilisant une requête vide
        results = self.vector_store.similarity_search("", k=100)
        documents = [doc.page_content for doc in results]
        
        # Nettoyage des retours
        documents = [doc.replace("\n", " ") for doc in documents]
        return documents
        
    # def clear(self) -> None:
    #     """
    #     Supprime toutes les données du vector store
    #     """
    #     if os.path.exists(self.persist_dir):
    #         shutil.rmtree(self.persist_dir)
    #         os.makedirs(self.persist_dir)
    #     self.vector_store = None
    def clear(self) -> None:
        """
        Supprime toutes les données du vector store
        """
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir)
        # self.vector_store = None
        logging.info("Vector store cleared.")

    async def get_context(self) -> str:
        """
        Retrieve context for RAG.
        """
        if not self.vector_store:
            logging.error("Vector store not initialized.")
            raise ValueError("Vector store not initialized. Please add documents first.")

        try:
            # Perform a similarity search with an empty query to get the most relevant documents
            results = self.vector_store.similarity_search("", k=5)
            if not results:
                logging.info("No documents found in similarity search.")
                return ""

            context = "\n\n".join([doc.page_content for doc in results])
            logging.info(f"Generated context: {context}")
            return context

        except Exception as e:
            logging.error(f"Error retrieving context from vector store: {str(e)}")
            raise
