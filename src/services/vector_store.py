"""Vector store service for managing embeddings and similarity search."""
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import os

import chromadb
from chromadb.config import Settings as ChromaSettings
import faiss
import numpy as np
import pickle

from ..config import settings
from ..core.exceptions import VectorStoreError, ConfigurationError
from ..schemas.document import DocumentChunkResponse


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def store_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Store embeddings with metadata."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, embedding_ids: List[str]) -> bool:
        """Delete embeddings by IDs."""
        pass
    
    @abstractmethod
    async def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding by ID."""
        pass
    
    @abstractmethod
    async def update_metadata(
        self,
        embedding_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update embedding metadata."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self):
        self.collection_name = settings.vector_db.collection_name
        
        # Initialize ChromaDB client
        if settings.vector_db.chroma_host:
            # Remote ChromaDB
            self.client = chromadb.HttpClient(
                host=settings.vector_db.chroma_host,
                port=settings.vector_db.chroma_port
            )
        else:
            # Local ChromaDB
            os.makedirs(settings.vector_db.chroma_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=settings.vector_db.chroma_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    async def store_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Store embeddings in ChromaDB."""
        try:
            if len(chunks) != len(embeddings):
                raise VectorStoreError("Chunks and embeddings count mismatch")
            
            # Prepare data for ChromaDB
            embedding_ids = [str(uuid.uuid4()) for _ in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "document_id": str(chunk["document_id"]),
                    "document_name": chunk.get("document_name", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title", ""),
                    "tags": chunk.get("tags", []),
                    "source": chunk.get("source", ""),
                }
                for chunk in chunks
            ]
            
            # Add to collection
            self.collection.add(
                ids=embedding_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            return embedding_ids
        
        except Exception as e:
            raise VectorStoreError(f"Failed to store embeddings: {str(e)}")
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform similarity search in ChromaDB."""
        try:
            # Convert filters to ChromaDB format
            where_filters = None
            if filters:
                where_filters = {}
                
                if "document_ids" in filters:
                    where_filters["document_id"] = {"$in": filters["document_ids"]}
                
                if "tags" in filters and filters["tags"]:
                    where_filters["tags"] = {"$in": filters["tags"]}
                
                if "source" in filters:
                    where_filters["source"] = {"$eq": filters["source"]}
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # Convert distance to similarity score (cosine similarity)
                similarity_score = 1.0 - distance
                
                if similarity_score >= score_threshold:
                    formatted_results.append({
                        "chunk_id": metadata["chunk_id"],
                        "document_id": metadata["document_id"],
                        "document_name": metadata["document_name"],
                        "content": doc,
                        "score": similarity_score,
                        "chunk_index": metadata.get("chunk_index", 0),
                        "page_number": metadata.get("page_number"),
                        "section_title": metadata.get("section_title", ""),
                        "metadata": metadata,
                    })
            
            return formatted_results
        
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {str(e)}")
    
    async def delete_embeddings(self, embedding_ids: List[str]) -> bool:
        """Delete embeddings from ChromaDB."""
        try:
            self.collection.delete(ids=embedding_ids)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete embeddings: {str(e)}")
    
    async def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding by ID from ChromaDB."""
        try:
            results = self.collection.get(
                ids=[embedding_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not results["ids"]:
                return None
            
            return {
                "id": results["ids"][0],
                "document": results["documents"][0],
                "metadata": results["metadatas"][0],
                "embedding": results["embeddings"][0]
            }
        
        except Exception as e:
            raise VectorStoreError(f"Failed to get embedding: {str(e)}")
    
    async def update_metadata(
        self,
        embedding_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update embedding metadata in ChromaDB."""
        try:
            self.collection.update(
                ids=[embedding_id],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to update metadata: {str(e)}")


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self):
        self.dimension = 1536  # Default OpenAI embedding dimension
        self.index_file = os.path.join(settings.vector_db.chroma_path, "faiss.index")
        self.metadata_file = os.path.join(settings.vector_db.chroma_path, "metadata.pkl")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        
        # Initialize or load FAISS index
        self.index = None
        self.id_to_metadata = {}
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
            else:
                # Create new index (using IndexFlatIP for cosine similarity)
                self.index = faiss.IndexFlatIP(self.dimension)
            
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    self.id_to_metadata = pickle.load(f)
        
        except Exception as e:
            raise VectorStoreError(f"Failed to load FAISS index: {str(e)}")
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        try:
            faiss.write_index(self.index, self.index_file)
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.id_to_metadata, f)
        
        except Exception as e:
            raise VectorStoreError(f"Failed to save FAISS index: {str(e)}")
    
    async def store_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Store embeddings in FAISS."""
        try:
            if len(chunks) != len(embeddings):
                raise VectorStoreError("Chunks and embeddings count mismatch")
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(embeddings_array)
            
            # Store metadata
            embedding_ids = []
            for i, chunk in enumerate(chunks):
                embedding_id = str(uuid.uuid4())
                faiss_id = start_idx + i
                
                self.id_to_metadata[embedding_id] = {
                    "faiss_id": faiss_id,
                    "chunk_id": str(chunk["chunk_id"]),
                    "document_id": str(chunk["document_id"]),
                    "document_name": chunk.get("document_name", ""),
                    "content": chunk["content"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title", ""),
                    "tags": chunk.get("tags", []),
                    "source": chunk.get("source", ""),
                }
                
                embedding_ids.append(embedding_id)
            
            # Save index
            self._save_index()
            
            return embedding_ids
        
        except Exception as e:
            raise VectorStoreError(f"Failed to store embeddings: {str(e)}")
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform similarity search in FAISS."""
        try:
            # Normalize query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, k * 2)  # Get more for filtering
            
            # Format results with metadata filtering
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                # Find metadata by FAISS ID
                metadata = None
                for emb_id, meta in self.id_to_metadata.items():
                    if meta["faiss_id"] == idx:
                        metadata = meta
                        break
                
                if metadata is None:
                    continue
                
                # Apply filters
                if filters:
                    if "document_ids" in filters:
                        if metadata["document_id"] not in filters["document_ids"]:
                            continue
                    
                    if "tags" in filters and filters["tags"]:
                        if not any(tag in metadata["tags"] for tag in filters["tags"]):
                            continue
                    
                    if "source" in filters:
                        if metadata["source"] != filters["source"]:
                            continue
                
                # Check score threshold
                if score >= score_threshold:
                    results.append({
                        "chunk_id": metadata["chunk_id"],
                        "document_id": metadata["document_id"],
                        "document_name": metadata["document_name"],
                        "content": metadata["content"],
                        "score": float(score),
                        "chunk_index": metadata["chunk_index"],
                        "page_number": metadata["page_number"],
                        "section_title": metadata["section_title"],
                        "metadata": metadata,
                    })
                
                if len(results) >= k:
                    break
            
            return results
        
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {str(e)}")
    
    async def delete_embeddings(self, embedding_ids: List[str]) -> bool:
        """Delete embeddings from FAISS (not supported directly)."""
        # FAISS doesn't support deletion, would need to rebuild index
        # For now, just remove from metadata
        try:
            for embedding_id in embedding_ids:
                if embedding_id in self.id_to_metadata:
                    del self.id_to_metadata[embedding_id]
            
            self._save_index()
            return True
        
        except Exception as e:
            raise VectorStoreError(f"Failed to delete embeddings: {str(e)}")
    
    async def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding by ID from FAISS."""
        if embedding_id in self.id_to_metadata:
            return self.id_to_metadata[embedding_id]
        return None
    
    async def update_metadata(
        self,
        embedding_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update embedding metadata in FAISS."""
        try:
            if embedding_id in self.id_to_metadata:
                self.id_to_metadata[embedding_id].update(metadata)
                self._save_index()
                return True
            return False
        
        except Exception as e:
            raise VectorStoreError(f"Failed to update metadata: {str(e)}")


class VectorStoreService:
    """Main vector store service."""
    
    def __init__(self):
        self.provider = settings.vector_db.provider
        self.store = self._create_store()
    
    def _create_store(self) -> VectorStore:
        """Create vector store based on configuration."""
        if self.provider == "chroma":
            return ChromaVectorStore()
        elif self.provider == "faiss":
            return FAISSVectorStore()
        else:
            raise ConfigurationError(f"Unsupported vector store provider: {self.provider}")
    
    async def store_chunks_with_embeddings(
        self,
        chunks: List[DocumentChunkResponse],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Store document chunks with their embeddings."""
        # Convert chunks to dict format
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "tags": chunk.metadata.get("tags", []) if chunk.metadata else [],
                "source": chunk.metadata.get("source", "") if chunk.metadata else "",
            }
            chunk_dicts.append(chunk_dict)
        
        return await self.store.store_embeddings(chunk_dicts, embeddings)
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        return await self.store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            score_threshold=score_threshold
        )
    
    async def delete_document_embeddings(self, document_id: uuid.UUID) -> bool:
        """Delete all embeddings for a document."""
        # This would require getting all embedding IDs for the document first
        # Implementation depends on the specific vector store capabilities
        return True
    
    async def update_chunk_metadata(
        self,
        embedding_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update chunk metadata."""
        return await self.store.update_metadata(embedding_id, metadata)


# Global vector store service instance
vector_store_service = VectorStoreService()