"""Document processing service."""
import uuid
import hashlib
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import mimetypes

import PyPDF2
import pandas as pd
from bs4 import BeautifulSoup
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import settings
from ..models.document import Document, DocumentChunk
from ..models.user import User
from ..schemas.document import DocumentCreate, IngestRequest
from ..core.exceptions import ProcessingError, ValidationError
from ..core.logging import BusinessLogger
from .llm import llm_service
from .vector_store import vector_store_service

logger = BusinessLogger()


class DocumentProcessor:
    """Document processing utilities."""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ProcessingError(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_html(content: str) -> str:
        """Extract text from HTML content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            raise ProcessingError(f"Failed to extract text from HTML: {str(e)}")
    
    @staticmethod
    def extract_text_from_csv(file_path: str) -> str:
        """Extract text from CSV file."""
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to text representation
            text = df.to_string(index=False)
            return text
        except Exception as e:
            raise ProcessingError(f"Failed to extract text from CSV: {str(e)}")
    
    @staticmethod
    def extract_text_from_json(file_path: str) -> str:
        """Extract text from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = file.read()
            
            # Try to parse as JSON and format nicely
            try:
                import json
                parsed = json.loads(data)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # Return raw content if not valid JSON
                return data
        except Exception as e:
            raise ProcessingError(f"Failed to extract text from JSON: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ProcessingError("Could not decode text file with any supported encoding")
        except Exception as e:
            raise ProcessingError(f"Failed to extract text from file: {str(e)}")
    
    @staticmethod
    async def extract_text_from_url(url: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from URL."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                metadata = {
                    'url': url,
                    'content_type': content_type,
                    'status_code': response.status_code,
                    'content_length': len(response.content)
                }
                
                if 'text/html' in content_type:
                    text = DocumentProcessor.extract_text_from_html(response.text)
                elif 'application/json' in content_type:
                    text = response.text
                elif 'text/plain' in content_type:
                    text = response.text
                else:
                    # Try to extract as text anyway
                    text = response.text
                
                return text, metadata
        
        except Exception as e:
            raise ProcessingError(f"Failed to extract text from URL: {str(e)}")
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text into smaller segments."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,
            )
            
            chunks = text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                chunk_objects.append({
                    'index': i,
                    'content': chunk,
                    'content_hash': chunk_hash,
                    'char_count': len(chunk),
                    'token_count': len(chunk.split()),  # Rough estimate
                })
            
            return chunk_objects
        
        except Exception as e:
            raise ProcessingError(f"Failed to chunk text: {str(e)}")
    
    @staticmethod
    def detect_file_type(filename: str, content: bytes = None) -> str:
        """Detect file type from filename and optionally content."""
        # Get MIME type from filename
        mime_type, _ = mimetypes.guess_type(filename)
        
        # Map MIME types to our file types
        if mime_type:
            if mime_type.startswith('text/'):
                if filename.lower().endswith('.md'):
                    return 'markdown'
                elif filename.lower().endswith('.csv'):
                    return 'csv'
                else:
                    return 'text'
            elif mime_type == 'application/pdf':
                return 'pdf'
            elif mime_type == 'application/json':
                return 'json'
            elif mime_type.startswith('text/html'):
                return 'html'
        
        # Fallback to extension-based detection
        ext = Path(filename).suffix.lower()
        extension_map = {
            '.pdf': 'pdf',
            '.txt': 'text',
            '.md': 'markdown',
            '.csv': 'csv',
            '.json': 'json',
            '.html': 'html',
            '.htm': 'html',
        }
        
        return extension_map.get(ext, 'text')


class DocumentService:
    """Document management service."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def ingest_file(
        self,
        db: AsyncSession,
        file_path: str,
        filename: str,
        user: User,
        ingest_request: IngestRequest
    ) -> Document:
        """Ingest a file and process it."""
        try:
            # Create document record
            file_stats = Path(file_path).stat()
            file_type = self.processor.detect_file_type(filename)
            mime_type, _ = mimetypes.guess_type(filename)
            
            document = Document(
                filename=f"{uuid.uuid4()}_{filename}",
                original_filename=filename,
                file_type=file_type,
                file_size=file_stats.st_size,
                mime_type=mime_type,
                file_path=file_path,
                processing_status="pending",
                metadata=ingest_request.metadata,
                tags=ingest_request.tags,
                uploaded_by=user.id,
            )
            
            db.add(document)
            await db.commit()
            await db.refresh(document)
            
            # Process document asynchronously
            asyncio.create_task(
                self._process_document_async(db, document, ingest_request)
            )
            
            logger.log_document_uploaded(
                document_id=str(document.id),
                filename=filename,
                file_size=file_stats.st_size,
                user_id=str(user.id),
                processing_options=ingest_request.processing_options
            )
            
            return document
        
        except Exception as e:
            raise ProcessingError(f"Failed to ingest file: {str(e)}")
    
    async def ingest_url(
        self,
        db: AsyncSession,
        url: str,
        user: User,
        ingest_request: IngestRequest
    ) -> Document:
        """Ingest content from URL."""
        try:
            # Extract text from URL
            text_content, url_metadata = await self.processor.extract_text_from_url(url)
            
            # Create document record
            document = Document(
                filename=f"url_{uuid.uuid4()}.html",
                original_filename=url.split('/')[-1] or "webpage",
                file_type="url",
                file_size=len(text_content.encode()),
                mime_type=url_metadata.get('content_type'),
                source_url=url,
                processing_status="pending",
                metadata={**ingest_request.metadata, **url_metadata},
                tags=ingest_request.tags,
                uploaded_by=user.id,
            )
            
            db.add(document)
            await db.commit()
            await db.refresh(document)
            
            # Process document with extracted content
            asyncio.create_task(
                self._process_document_with_content(
                    db, document, text_content, ingest_request
                )
            )
            
            logger.log_document_uploaded(
                document_id=str(document.id),
                filename=document.original_filename,
                file_size=document.file_size,
                user_id=str(user.id),
                processing_options=ingest_request.processing_options
            )
            
            return document
        
        except Exception as e:
            raise ProcessingError(f"Failed to ingest URL: {str(e)}")
    
    async def _process_document_async(
        self,
        db: AsyncSession,
        document: Document,
        ingest_request: IngestRequest
    ):
        """Process document asynchronously."""
        start_time = datetime.utcnow()
        
        try:
            # Update status
            await db.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(processing_status="processing")
            )
            await db.commit()
            
            # Extract text based on file type
            if document.file_type == "pdf":
                text_content = self.processor.extract_text_from_pdf(document.file_path)
            elif document.file_type == "csv":
                text_content = self.processor.extract_text_from_csv(document.file_path)
            elif document.file_type == "json":
                text_content = self.processor.extract_text_from_json(document.file_path)
            elif document.file_type in ["text", "markdown", "html"]:
                text_content = self.processor.extract_text_from_txt(document.file_path)
            else:
                text_content = self.processor.extract_text_from_txt(document.file_path)
            
            await self._process_document_with_content(
                db, document, text_content, ingest_request
            )
            
        except Exception as e:
            # Update status to failed
            await db.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(
                    processing_status="failed",
                    processing_error=str(e)
                )
            )
            await db.commit()
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.log_document_processed(
                document_id=str(document.id),
                filename=document.original_filename,
                chunks_created=0,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _process_document_with_content(
        self,
        db: AsyncSession,
        document: Document,
        text_content: str,
        ingest_request: IngestRequest
    ):
        """Process document with extracted text content."""
        start_time = datetime.utcnow()
        
        try:
            # Get processing options
            options = ingest_request.processing_options
            chunk_size = options.get('chunk_size', 800)
            chunk_overlap = options.get('chunk_overlap', 100)
            
            # Chunk the text
            chunks = self.processor.chunk_text(
                text_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if not chunks:
                raise ProcessingError("No chunks created from document")
            
            # Create chunk records
            chunk_records = []
            for chunk_data in chunks:
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk_data['index'],
                    content=chunk_data['content'],
                    content_hash=chunk_data['content_hash'],
                    token_count=chunk_data['token_count'],
                    char_count=chunk_data['char_count'],
                    metadata={}
                )
                chunk_records.append(chunk_record)
            
            # Add chunks to database
            db.add_all(chunk_records)
            await db.commit()
            
            # Refresh to get IDs
            for chunk in chunk_records:
                await db.refresh(chunk)
            
            # Generate embeddings
            embedding_model = options.get('embedding_model', 'openai')
            chunk_texts = [chunk.content for chunk in chunk_records]
            embeddings = await llm_service.batch_embeddings(
                texts=chunk_texts,
                provider=embedding_model
            )
            
            # Store embeddings in vector database
            from ..schemas.document import DocumentChunkResponse
            chunk_responses = [
                DocumentChunkResponse.model_validate(chunk)
                for chunk in chunk_records
            ]
            
            embedding_ids = await vector_store_service.store_chunks_with_embeddings(
                chunks=chunk_responses,
                embeddings=embeddings
            )
            
            # Update chunks with embedding IDs
            for chunk, embedding_id in zip(chunk_records, embedding_ids):
                chunk.embedding_id = embedding_id
            
            await db.commit()
            
            # Update document status
            await db.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(
                    processing_status="completed",
                    total_chunks=len(chunks),
                    total_tokens=sum(chunk['token_count'] for chunk in chunks)
                )
            )
            await db.commit()
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.log_document_processed(
                document_id=str(document.id),
                filename=document.original_filename,
                chunks_created=len(chunks),
                processing_time_ms=processing_time,
                success=True
            )
        
        except Exception as e:
            # Update status to failed
            await db.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(
                    processing_status="failed",
                    processing_error=str(e)
                )
            )
            await db.commit()
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.log_document_processed(
                document_id=str(document.id),
                filename=document.original_filename,
                chunks_created=0,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def get_document_by_id(
        self,
        db: AsyncSession,
        document_id: uuid.UUID,
        user: User = None
    ) -> Optional[Document]:
        """Get document by ID."""
        stmt = select(Document).where(Document.id == document_id)
        
        # Add user filter for non-admin users
        if user and user.role not in ["admin"]:
            stmt = stmt.where(Document.uploaded_by == user.id)
        
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_documents(
        self,
        db: AsyncSession,
        user: User,
        skip: int = 0,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], int]:
        """Get documents with pagination and filtering."""
        stmt = select(Document)
        
        # Add user filter for non-admin users
        if user.role not in ["admin"]:
            stmt = stmt.where(Document.uploaded_by == user.id)
        
        # Apply filters
        if filters:
            if "file_types" in filters:
                stmt = stmt.where(Document.file_type.in_(filters["file_types"]))
            
            if "status" in filters:
                stmt = stmt.where(Document.processing_status.in_(filters["status"]))
            
            if "tags" in filters:
                for tag in filters["tags"]:
                    stmt = stmt.where(Document.tags.any(tag))
        
        # Get total count
        count_stmt = select(Document.id).select_from(stmt.subquery())
        count_result = await db.execute(count_stmt)
        total = len(count_result.all())
        
        # Apply pagination
        stmt = stmt.offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        documents = result.scalars().all()
        
        return list(documents), total
    
    async def delete_document(
        self,
        db: AsyncSession,
        document_id: uuid.UUID,
        user: User
    ) -> bool:
        """Delete document and all associated chunks."""
        document = await self.get_document_by_id(db, document_id, user)
        if not document:
            return False
        
        # Delete from vector store (if implemented)
        try:
            await vector_store_service.delete_document_embeddings(document_id)
        except Exception as e:
            # Log but don't fail the deletion
            print(f"Warning: Failed to delete vector embeddings: {e}")
        
        # Delete document (chunks will be deleted due to cascade)
        await db.delete(document)
        await db.commit()
        
        # Delete file if it exists
        if document.file_path and Path(document.file_path).exists():
            try:
                Path(document.file_path).unlink()
            except Exception as e:
                print(f"Warning: Failed to delete file {document.file_path}: {e}")
        
        return True


# Global document service instance
document_service = DocumentService()