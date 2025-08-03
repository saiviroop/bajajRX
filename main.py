# main.py - HackRx 6.0 Complete Implementation
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
import time
import os
import hashlib
import json
import re
from dataclasses import dataclass

# Required imports (install via requirements.txt)
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyMuPDF  # fitz
from docx import Document as DocxDocument
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="HackRx 6.0 Document Processor",
    description="LLM-powered intelligent query-retrieval system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class ProcessingMetadata(BaseModel):
    processing_time: float
    document_size: int
    questions_processed: int
    confidence_scores: List[float]

class HackRxResponse(BaseModel):
    answers: List[str]
    metadata: Optional[ProcessingMetadata] = None

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    source_info: Optional[str] = None

class AdvancedDocumentProcessor:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Document cache
        self.doc_cache = {}
        
        # Insurance domain terms for query enhancement
        self.insurance_terms = {
            'grace period': ['payment grace', 'premium grace', 'renewal grace', 'due date'],
            'waiting period': ['coverage waiting', 'benefit waiting', 'claim waiting', 'PED waiting'],
            'pre-existing': ['PED', 'pre-existing disease', 'existing condition', 'prior condition'],
            'maternity': ['pregnancy', 'childbirth', 'delivery', 'maternal', 'obstetric'],
            'AYUSH': ['Ayurveda', 'Yoga', 'Naturopathy', 'Unani', 'Siddha', 'Homeopathy', 'alternative medicine'],
            'NCD': ['No Claim Discount', 'no claim bonus', 'bonus', 'discount'],
            'hospital': ['medical facility', 'healthcare facility', 'clinic', 'medical institution']
        }
        
    async def download_document(self, url: str) -> bytes:
        """Download document from URL with caching"""
        # Create cache key
        cache_key = hashlib.md5(url.encode()).hexdigest()
        
        if cache_key in self.doc_cache:
            logger.info(f"Using cached document: {cache_key}")
            return self.doc_cache[cache_key]
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download document: {response.status}")
                
                content = await response.read()
                self.doc_cache[cache_key] = content
                logger.info(f"Downloaded and cached document: {len(content)} bytes")
                return content

    def extract_text_from_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = PyMuPDF.open(stream=pdf_content, filetype="pdf")
            full_text = ""
            page_texts = []
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts.append({
                    'page': page_num + 1,
                    'text': page_text
                })
                full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
            
            doc.close()
            
            return {
                'text': full_text,
                'pages': page_texts,
                'page_count': len(page_texts),
                'format': 'pdf'
            }
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

    def extract_text_from_docx(self, docx_content: bytes) -> Dict[str, Any]:
        """Extract text from DOCX"""
        try:
            doc = DocxDocument(io.BytesIO(docx_content))
            paragraphs = []
            full_text = ""
            
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
                    full_text += para.text + "\n"
            
            return {
                'text': full_text,
                'paragraphs': paragraphs,
                'paragraph_count': len(paragraphs),
                'format': 'docx'
            }
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"DOCX processing failed: {str(e)}")

    def detect_format_and_extract(self, content: bytes, url: str) -> Dict[str, Any]:
        """Detect document format and extract text"""
        # Simple format detection based on URL and content
        if url.lower().endswith('.pdf') or content.startswith(b'%PDF'):
            return self.extract_text_from_pdf(content)
        elif url.lower().endswith('.docx') or b'word/' in content[:1000]:
            return self.extract_text_from_docx(content)
        else:
            # Try PDF first, then DOCX
            try:
                return self.extract_text_from_pdf(content)
            except:
                try:
                    return self.extract_text_from_docx(content)
                except:
                    # Fallback to plain text
                    try:
                        text = content.decode('utf-8')
                        return {'text': text, 'format': 'text'}
                    except:
                        raise HTTPException(status_code=400, detail="Unsupported document format")

    def intelligent_chunking(self, doc_data: Dict[str, Any]) -> List[DocumentChunk]:
        """Create intelligent chunks preserving context"""
        text = doc_data['text']
        chunks = []
        
        # Split by sections first (common in policy documents)
        section_patterns = [
            r'\n\s*(?:SECTION|Section|Article|ARTICLE)\s+[\d\w\.]+.*?\n',
            r'\n\s*\d+\.\s+[A-Z][^.]*\n',
            r'\n\s*[A-Z][A-Z\s]{10,}\n',  # All caps headers
        ]
        
        sections = []
        current_text = text
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
            if matches:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(current_text)
                    section_text = current_text[start:end].strip()
                    if len(section_text) > 100:  # Minimum section size
                        sections.append(section_text)
                break
        
        # If no sections found, use paragraph-based chunking
        if not sections:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            sections = paragraphs
        
        # Create chunks with overlap
        chunk_size = 800
        overlap = 150
        
        for i, section in enumerate(sections):
            if len(section) <= chunk_size:
                # Small section, use as-is
                chunks.append(DocumentChunk(
                    text=section,
                    metadata={
                        'chunk_id': f"section_{i}",
                        'section_number': i,
                        'type': 'section',
                        'length': len(section)
                    }
                ))
            else:
                # Large section, split with overlap
                words = section.split()
                for j in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[j:j + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        metadata={
                            'chunk_id': f"section_{i}_part_{j//chunk_size}",
                            'section_number': i,
                            'part_number': j // chunk_size,
                            'type': 'section_part',
                            'length': len(chunk_text)
                        }
                    ))
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks

    def enhance_query_for_insurance(self, query: str) -> List[str]:
        """Enhance query with insurance-specific terms"""
        enhanced_queries = [query]  # Original query
        
        # Add variations with insurance terminology
        query_lower = query.lower()
        
        for term, synonyms in self.insurance_terms.items():
            if any(syn.lower() in query_lower for syn in [term] + synonyms):
                # Create variations
                for synonym in synonyms:
                    if synonym.lower() not in query_lower:
                        enhanced_query = query + f" {synonym}"
                        enhanced_queries.append(enhanced_query)
        
        # Add common insurance question patterns
        if 'cover' in query_lower or 'coverage' in query_lower:
            enhanced_queries.append(f"policy coverage {query}")
            enhanced_queries.append(f"benefits {query}")
        
        if 'waiting' in query_lower or 'period' in query_lower:
            enhanced_queries.append(f"waiting period {query}")
            enhanced_queries.append(f"eligibility {query}")
        
        return enhanced_queries[:5]  # Limit to avoid too many queries

    def retrieve_relevant_chunks(self, question: str, chunks: List[DocumentChunk], top_k: int = 8) -> List[DocumentChunk]:
        """Multi-stage retrieval with re-ranking"""
        
        # Stage 1: Semantic similarity search
        enhanced_queries = self.enhance_query_for_insurance(question)
        
        all_scores = []
        for query in enhanced_queries:
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Calculate similarities
            chunk_embeddings = np.array([chunk.embedding for chunk in chunks])
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            all_scores.append(similarities)
        
        # Combine scores (max pooling)
        combined_scores = np.max(all_scores, axis=0)
        
        # Stage 2: Keyword boosting
        question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
        
        for i, chunk in enumerate(chunks):
            chunk_keywords = set(re.findall(r'\b\w+\b', chunk.text.lower()))
            keyword_overlap = len(question_keywords.intersection(chunk_keywords))
            keyword_boost = min(keyword_overlap * 0.1, 0.3)  # Max 30% boost
            combined_scores[i] += keyword_boost
        
        # Get top chunks
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        # Add relevance scores to metadata
        for chunk, idx in zip(relevant_chunks, top_indices):
            chunk.metadata['relevance_score'] = float(combined_scores[idx])
        
        return relevant_chunks

    def format_chunks_for_prompt(self, chunks: List[DocumentChunk]) -> str:
        """Format chunks for LLM prompt with references"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.text.strip()
            metadata = chunk.metadata
            
            reference = f"[Chunk {i}]"
            if 'section_number' in metadata:
                reference += f" Section {metadata['section_number']}"
            if 'page' in metadata:
                reference += f" (Page {metadata['page']})"
            
            formatted_chunk = f"{reference}:\n{chunk_text}\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n---\n".join(formatted_chunks)

    async def generate_answer_with_llm(self, question: str, relevant_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Generate answer using optimized LLM prompt"""
        
        formatted_chunks = self.format_chunks_for_prompt(relevant_chunks)
        
        # Optimized prompt for insurance/policy questions
        prompt = f"""You are an expert insurance policy analyst. Answer the question based ONLY on the provided policy clauses.

QUESTION: {question}

POLICY CLAUSES:
{formatted_chunks}

INSTRUCTIONS:
1. Answer the question precisely and directly
2. If it's about coverage: specify Yes/No and exact conditions
3. If it's about amounts: provide specific numbers, percentages, or limits
4. If it's about definitions: provide the exact definition from the policy
5. If it's about waiting periods: specify exact duration and conditions
6. Always reference the specific chunk/section where you found the information
7. Use exact wording from the policy when possible
8. If information is not available in the provided clauses, say so clearly

ANSWER FORMAT:
Provide a clear, direct answer that addresses the question completely. Include specific details like amounts, timeframes, and conditions when relevant.

Answer:"""

        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=400,   # Balanced for detailed answers
                timeout=25        # Ensure response under 30s
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on relevance scores
            avg_relevance = np.mean([chunk.metadata.get('relevance_score', 0) for chunk in relevant_chunks])
            confidence = min(avg_relevance * 1.2, 1.0)  # Scale and cap at 1.0
            
            return {
                'answer': answer,
                'confidence': confidence,
                'tokens_used': response.usage.total_tokens,
                'chunks_used': len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            return {
                'answer': f"Error processing question: {str(e)}",
                'confidence': 0.0,
                'tokens_used': 0,
                'chunks_used': 0
            }

    async def process_questions_batch(self, questions: List[str], chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Process multiple questions in parallel"""
        
        async def process_single_question(question: str) -> Dict[str, Any]:
            # Retrieve relevant chunks for this question
            relevant_chunks = self.retrieve_relevant_chunks(question, chunks)
            
            # Generate answer
            result = await self.generate_answer_with_llm(question, relevant_chunks)
            
            return result
        
        # Process all questions in parallel
        results = await asyncio.gather(*[process_single_question(q) for q in questions])
        
        return results

# Global processor instance
processor = AdvancedDocumentProcessor()

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify Bearer token"""
    expected_token = "6f1f341508f756f9e85ac3beeccbe53ab1808a2a650b81c04abeaa80f81356d7"
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HackRx 6.0 Document Processor API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post("/hackrx/run", response_model=HackRxResponse)
async def process_hackrx_request(
    request: HackRxRequest,
    token: str = Depends(verify_token)
) -> HackRxResponse:
    """Main endpoint for processing documents and questions"""
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Step 1: Download and extract document
        doc_content = await processor.download_document(request.documents)
        doc_data = processor.detect_format_and_extract(doc_content, request.documents)
        
        logger.info(f"Extracted document: {len(doc_data['text'])} characters")
        
        # Step 2: Create intelligent chunks
        chunks = processor.intelligent_chunking(doc_data)
        
        # Step 3: Process all questions
        results = await processor.process_questions_batch(request.questions, chunks)
        
        # Step 4: Extract answers and metadata
        answers = [result['answer'] for result in results]
        confidence_scores = [result['confidence'] for result in results]
        
        processing_time = time.time() - start_time
        
        metadata = ProcessingMetadata(
            processing_time=processing_time,
            document_size=len(doc_data['text']),
            questions_processed=len(request.questions),
            confidence_scores=confidence_scores
        )
        
        logger.info(f"Completed processing in {processing_time:.2f}s")
        
        return HackRxResponse(
            answers=answers,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cache_size": len(processor.doc_cache),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gpt-4"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))