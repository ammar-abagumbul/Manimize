from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown
from urllib.parse import urljoin, urlparse
import requests
import time
from typing import Generator, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Generator
import json
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import tiktoken
import re
import copy

# Configuration constants
BASE_URL = "https://docs.manim.community/en/latest/"
STARTING_PAGE = "reference_index/animations.html"
CRAWL_DEPTH = 3
REQUEST_DELAY = 0.5  # Delay between requests in seconds


class WebCrawler:
    """Handles web crawling and data extraction"""
    
    def __init__(self, rate_limit: float = 0.5, timeout: int = 10):
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.target_tags = []
        self.stopping_condition = None
    
    def set_target_tags(self, tags: List[str]) -> None:
        self.target_tags = tags
    
    def set_stopping_condition(self, callback: callable = None):
        self.stopping_condition = callback
    
    def crawl(self, url: str, depth: int = 10) -> Generator[Dict, None, None]:
        current_url = url
        pages_crawled = 0
        
        while pages_crawled < depth and current_url:
            if self.stopping_condition and not self.stopping_condition(current_url):
                print(f"Stopping condition met for URL: {current_url}")
                break
            try:
                print(f"Crawling: {current_url}")
                response = requests.get(current_url, timeout=self.timeout)
                
                if response.status_code != 200:
                    print(f"Failed to retrieve {current_url}: HTTP {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')

                next_link = self._get_next_link(soup)
                page_data = self._extract_page_data(current_url, soup)

                yield page_data
                
                pages_crawled += 1
                
                if next_link:
                    current_url = urljoin(current_url, next_link)
                else:
                    print("No more pages found")
                    break
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                print(f"Error crawling {current_url}: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"Crawling completed. Total pages: {pages_crawled}")
    
    def _get_next_link(self, soup: BeautifulSoup) -> Optional[str]:
        next_link = soup.find('a', class_='next-page')
        if next_link and 'href' in next_link.attrs:
            return next_link['href']
        return None
    
    def _extract_page_data(self, url: str, soup: BeautifulSoup) -> Dict:
        """Extract relevant content from soup"""

        soup = copy.deepcopy(soup)
        target_tags = self.target_tags

        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        markdown = []
        if target_tags:
            tags = soup(target_tags)
            for tag in tags:
                raw = convert_to_markdown(tag) 
                processed = "\n".join(line for line in raw.splitlines() if line.strip() != '')
                markdown.append(processed)
        else: 
            raw = convert_to_markdown(tag) 
            processed = "\n".join(line for line in raw.splitlines() if line.strip() != '')
            markdown.append(processed)
        
        cleaned_text = "\n".join(markdown)

        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""

        # Generate document ID
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        return {
            'doc_id': doc_id,
            'url': url,
            'title': title_text,
            'content': cleaned_text,
            'scraped_at': datetime.now().isoformat(),
            'content_length': len(cleaned_text),
            'domain': urlparse(url).netloc,
            'word_count': len(cleaned_text.split())
        }

class DataInspector:
    """Handles manual inspection and file-based storage"""
    
    def __init__(self, inspection_dir: str = "./scraped_data"):
        self.inspection_dir = Path(inspection_dir)
        self.inspection_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.inspection_dir / "raw").mkdir(exist_ok=True)
        (self.inspection_dir / "approved").mkdir(exist_ok=True)
        (self.inspection_dir / "rejected").mkdir(exist_ok=True)
    
    def save_for_inspection(self, page_data: Dict) -> str:
        """Save scraped data into json files"""
        
        safe_url = self._safe_filename(page_data['url'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title = page_data['title']
        filename = f"{title}_{timestamp}.json"
        
        filepath = self.inspection_dir / "raw" / filename
        
        # Add inspection metadata
        inspection_data = {
            **page_data,
            'inspection_status': 'pending',
            'filename': filename,
            'saved_at': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(inspection_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved for inspection: {filepath}")
        return str(filepath)
    
    def save_all_for_inspection(self, crawler: WebCrawler, url: str, depth: int = 10) -> List[str]:
        """Crawl and save all pages for inspection"""
        saved_files = []
        
        for page_data in crawler.crawl(url, depth):
            filepath = self.save_for_inspection(page_data)
            saved_files.append(filepath)
        
        print(f"\nInspection Summary:")
        print(f"Total files saved: {len(saved_files)}")
        print(f"Location: {self.inspection_dir / 'raw'}")
        print(f"\nTo review files, check: {self.inspection_dir / 'raw'}")
        
        return saved_files
    
    def approve_file(self, filename: str) -> str:
        """Move file from raw to approved directory"""
        raw_path = self.inspection_dir / "raw" / filename
        approved_path = self.inspection_dir / "approved" / filename
        
        if raw_path.exists():
            # Update status in file
            with open(raw_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['inspection_status'] = 'approved'
            data['approved_at'] = datetime.now().isoformat()
            
            # Save to approved directory
            with open(approved_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Remove from raw
            raw_path.unlink()
            print(f"Approved: {filename}")
            return str(approved_path)
        else:
            raise FileNotFoundError(f"File not found: {filename}")
    
    def reject_file(self, filename: str, reason: str = "") -> str:
        """Move file from raw to rejected directory"""
        raw_path = self.inspection_dir / "raw" / filename
        rejected_path = self.inspection_dir / "rejected" / filename
        
        if raw_path.exists():
            # Update status in file
            with open(raw_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['inspection_status'] = 'rejected'
            data['rejected_at'] = datetime.now().isoformat()
            data['rejection_reason'] = reason
            
            # Save to rejected directory
            with open(rejected_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Remove from raw
            raw_path.unlink()
            print(f"Rejected: {filename} - {reason}")
            return str(rejected_path)
        else:
            raise FileNotFoundError(f"File not found: {filename}")
    
    def get_pending_files(self) -> List[str]:
        """Get list of files pending inspection"""
        raw_dir = self.inspection_dir / "raw"
        return [f.name for f in raw_dir.glob("*.json")]
    
    def get_approved_files(self) -> List[str]:
        """Get list of approved files"""
        approved_dir = self.inspection_dir / "approved"
        return [f.name for f in approved_dir.glob("*.json")]
    
    def preview_file(self, filename: str, max_content_length: int = 500) -> Dict:
        """Preview a file's content for inspection"""
        raw_path = self.inspection_dir / "raw" / filename
        
        if raw_path.exists():
            with open(raw_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            preview = {
                'filename': filename,
                'url': data['url'],
                'title': data['title'],
                'domain': data['domain'],
                'word_count': data['word_count'],
                'content_preview': data['content'][:max_content_length] + "..." if len(data['content']) > max_content_length else data['content'],
                'scraped_at': data['scraped_at']
            }
            return preview
        else:
            raise FileNotFoundError(f"File not found: {filename}")
    
    def preview_file_markdown(self, filename: str):
        """Preview the content of a file in a temporary file"""
        raw_path = self.inspection_dir / "approved" / filename
        if raw_path.exists():
            with open(raw_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            content = data.get('content', '')
            # Create a temporary markdown file
            temp_file = self.inspection_dir / "temp_preview.md"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(f"# {data['title']}\n\n")
                f.write(f"**URL:** {data['url']}\n\n")
                f.write(f"**Domain:** {data['domain']}\n\n")
                f.write(f"**Word Count:** {data['word_count']}\n\n")
                f.write(f"**Scraped At:** {data['scraped_at']}\n\n")
                f.write(content)
        else:
            raise FileNotFoundError(f"File not found: {filename}")
    
    def _safe_filename(self, url: str) -> str:
        """Convert URL to safe filename"""
        safe = url.replace('https://', '').replace('http://', '')
        safe = re.sub(r'[^\w\-_.]', '_', safe)
        return safe[:50] 

class RAGDataProcessor:
    """Handles RAG-specific processing and vector storage"""
    
    def __init__(self, 
                 chroma_path: str = "./chroma_db",
                 collection_name: str = "scraped_content",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Text processing settings
        # TODO: consider appropriate chunk size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def process_approved_files(self, inspector: DataInspector) -> int:
        """Process all approved files and store in vector database"""
        approved_files = inspector.get_approved_files()
        processed_count = 0
        
        for filename in approved_files:
            try:
                filepath = inspector.inspection_dir / "approved" / filename
                with open(filepath, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                
                self.process_and_store(page_data)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        print(f"Processed {processed_count} approved files")
        return processed_count
    
    def process_and_store(self, page_data: Dict) -> None:
        """Process page data and store in vector database"""
        
        # Skip if already processed
        existing = self.collection.get(ids=[page_data['doc_id']])
        if existing['ids']:
            print(f"Document {page_data['url']} already exists, skipping...")
            return
        
        # Prepare metadata for chunking
        base_metadata = {
            'doc_id': page_data['doc_id'],
            'url': page_data['url'],
            'title': page_data['title'],
            'domain': page_data['domain'],
            'scraped_at': page_data['scraped_at'],
            'word_count': page_data['word_count'],
        }
        # Chunk the text
        chunks = self._chunk_text(page_data['content'], base_metadata)
        
        if not chunks:
            print(f"No chunks generated for {page_data['url']}")
            return
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts).tolist()
        
        # Prepare data for storage
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Store in vector database
        self.collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Stored {len(chunks)} chunks for {page_data['url']}")
    
    def process_file(self, filepath: Path) -> None:
        """Process a single file and store in vector database"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
            self.process_and_store(page_data) 
        except Exception as e:
            print(f"Error processing file {filepath.name}: {str(e)}")
    
    def _chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into chunks suitable for RAG"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_metadata = {
                **metadata,
                'chunk_id': chunk_id,
                'chunk_start': start,
                'chunk_end': end,
                'chunk_length': len(chunk_text),
                'total_chunks': None
            }
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata,
                'chunk_id': f"{metadata['doc_id']}_chunk_{chunk_id}"
            })
            
            start = end - self.chunk_overlap
            chunk_id += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        """Search the vector database for relevant chunks"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_dict,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results

def main():
    inspector = DataInspector(inspection_dir="./manim_documentation") 
    file = inspector.get_approved_files()[0] 

    inspector.preview_file_markdown(file)


if __name__ == "__main__":
    main()