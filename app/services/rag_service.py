from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
import os
import shutil

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
        splits = []
        for text in texts:
            splits.extend(self.text_splitter.split_text(text))
        # splits = self.text_splitter.split_text(texts)
        
        if self.vector_store is None:
            # Création initiale
            self.vector_store = Chroma.from_texts(
                splits,
                self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            # Ajout à l'existant
            self.vector_store.add_texts(splits)
            
        # Persistance explicite
        self.vector_store.persist()
    
    async def similarity_search(self, query: str, k: int = 4) -> List[str]:
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
        return [doc.page_content for doc in results]
    
    def get_all_documents(self) -> List[str]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please add documents first.")
        
        return [doc.page_content for doc in self.vector_store.similarity_search("", k = 100)]
        
    def clear(self) -> None:
        """
        Supprime toutes les données du vector store
        """
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir)
        self.vector_store = None