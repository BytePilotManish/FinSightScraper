import requests
from bs4 import BeautifulSoup
import os
import json
import csv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import re
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import fitz  # PyMuPDF
import tempfile
import shutil
import random
from playwright_stealth import stealth_sync
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from datetime import datetime
import hashlib
from typing import Dict, List, Optional, Tuple
import concurrent.futures
from ratelimit import limits, sleep_and_retry
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

# Rate limiting configuration
CALLS = 3  # Number of calls
RATE_LIMIT_PERIOD = 60  # Time period in seconds

# Content type definitions
CONTENT_TYPES = {
    'definition': ['terms', 'definition', 'glossary'],
    'analysis': ['analysis', 'research', 'report'],
    'market': ['market', 'trading', 'investing'],
    'economic': ['gdp', 'inflation', 'unemployment', 'interest', 'cpi'],
    'investment': ['strategy', 'portfolio', 'diversification', 'allocation']
}

# Finance keywords for content filtering
FINANCE_KEYWORDS = [
    "stocks", "equity", "mutual funds", "ETF", "interest rates", "inflation", "RBI",
    "NSE", "BSE", "dividends", "portfolio", "investment", "banking", "fiscal",
    "SEBI", "FDI", "monetary policy", "GDP", "economic growth", "asset allocation",
    "bonds", "debt", "commodities", "crypto", "exchange rate", "market crash"
]

# Site-specific selectors
SITE_SELECTORS = {
    'reuters.com': {
        'title': ['h1', '.article-header__heading__3oVXM'],
        'content': ['.article-body__content__17Yit', '.article-body'],
        'date': ['time', '.article-header__timestamp__1DgxM'],
        'author': ['.author-name', '.article-header__author__2Ml1H']
    },
    'moneycontrol.com': {
        'title': ['h1', '.article_title'],
        'content': ['.article_content', '.content_wrapper'],
        'date': ['.article_time', '.date'],
        'author': ['.author', '.writer']
    },
    'livemint.com': {
        'title': ['h1', '.headline'],
        'content': ['.storyContent', '.mainContent'],
        'date': ['.dateTime', '.published'],
        'author': ['.author', '.writer']
    },
    'investopedia.com': {
        'title': ['h1', '.mntl-article-title', '.article-title'],
        'content': ['.mntl-sc-block', '.mntl-sc-block-group', '.article-body'],
        'date': ['time', '.mntl-attribution__item-date', '.article-date'],
        'author': ['.mntl-attribution__item-name', '.author', '.article-author'],
        'category': ['.mntl-breadcrumbs__item', '.article-category']
    },
    'rbi.org.in': {
        'title': ['h1', '.title'],
        'content': ['.content', '.report-content'],
        'date': ['.date', '.published-date'],
        'author': ['.author', '.department']
    },
    'sebi.gov.in': {
        'title': ['h1', '.title'],
        'content': ['.content', '.report-content'],
        'date': ['.date', '.published-date'],
        'author': ['.author', '.department']
    }
}

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_template(domain):
    with open(f"templates/{domain}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_urls(domain):
    """Load URLs from file, skipping comments and empty lines."""
    with open(f"urls/{domain}.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f 
                if line.strip() and not line.strip().startswith('#')]

def clean_text(text: str) -> str:
    """Clean and format text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def extract_pdf_text(url):
    """Extract text from PDF files."""
    try:
        # Download PDF to temporary file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file.flush()
            
            # Extract text using PyMuPDF
            doc = fitz.open(tmp_file.name)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
        # Clean up temporary file
        os.unlink(tmp_file.name)
        return clean_text(text)
    except Exception as e:
        print(f"[!] PDF extraction error for {url}: {e}")
        return None

def fetch_dynamic_html(url):
    """Fetch HTML content using Playwright with improved headers."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        
        try:
            # Set additional headers
            page.set_extra_http_headers({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0"
            })
            
            # Navigate with longer timeout
            page.goto(url, timeout=60000, wait_until="networkidle")
            
            # Wait for content to load
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_load_state("networkidle")
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            # Get the final HTML
            html = page.content()
            
            # Check for access denied
            if "access denied" in html.lower() or "blocked" in html.lower():
                print(f"[!] Access denied for {url}")
                return None
                
            return html
            
        except Exception as e:
            print(f"[!] Playwright error for {url}: {e}")
            return None
        finally:
            browser.close()

def extract_field(soup, selectors, fallback=True):
    """Extract text from HTML using selectors with fallback options."""
    # Try the provided selectors first
    for selector in selectors:
        elements = soup.select(selector)
        if elements:
            # For content blocks, combine all matching elements
            if selector.startswith('div.mntl-sc-block') or selector.startswith('div.mntl-sc-block-group'):
                text = ' '.join([clean_text(elem.get_text()) for elem in elements])
                if text:
                    return text
            # For titles, use the first matching element
            else:
                text = clean_text(elements[0].get_text())
                if text:
                    return text
    
    # Fallback strategies if no content found
    if fallback:
        # Try to find title in meta tags
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return clean_text(meta_title['content'])
        
        # Try to find content in article or main content areas
        for tag in ['article', 'main', 'div[role="main"]', 'div.content', 'div.article-content']:
            content = soup.select_one(tag)
            if content:
                text = clean_text(content.get_text())
                if text:
                    return text
        
        # Last resort: combine all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = ' '.join([clean_text(p.get_text()) for p in paragraphs])
            if text:
                return text
    
    return ""

def is_pdf_url(url):
    """Check if URL points to a PDF file."""
    return url.lower().endswith('.pdf')

def is_javascript_site(url):
    """Check if the URL is likely to require JavaScript rendering."""
    domain = urlparse(url).netloc.lower()
    js_sites = [
        'moneycontrol.com',
        'ndtv.com',
        'livemint.com',
        'economictimes.indiatimes.com',
        'reuters.com'
    ]
    return any(site in domain for site in js_sites)

def get_site_selectors(url: str) -> dict:
    """Get site-specific selectors based on URL."""
    domain = urlparse(url).netloc.lower()
    for site, selectors in SITE_SELECTORS.items():
        if site in domain:
            return selectors
    return {}

def extract_metadata(soup: BeautifulSoup, url: str) -> Dict:
    """Extract metadata from the page using site-specific selectors."""
    metadata = {
        'url': url,
        'title': '',
        'author': '',
        'date': '',
        'category': '',
        'content_type': '',
        'word_count': 0,
        'source': urlparse(url).netloc
    }
    
    # Get site-specific selectors
    site_selectors = get_site_selectors(url)
    
    # Extract title
    title_selectors = site_selectors.get('title', ['h1', 'meta[property="og:title"]', 'title'])
    for selector in title_selectors:
        element = soup.select_one(selector)
        if element:
            if selector == 'meta[property="og:title"]':
                metadata['title'] = element.get('content', '').strip()
            else:
                metadata['title'] = element.text.strip()
            if metadata['title']:
                break
    
    # Extract author
    author_selectors = site_selectors.get('author', [
        'meta[name="author"]',
        '.author',
        '[class*="author"]',
        '[class*="byline"]'
    ])
    for selector in author_selectors:
        element = soup.select_one(selector)
        if element:
            if selector == 'meta[name="author"]':
                metadata['author'] = element.get('content', '').strip()
            else:
                metadata['author'] = element.text.strip()
            if metadata['author']:
                break
    
    # Extract date
    date_selectors = site_selectors.get('date', [
        'meta[property="article:published_time"]',
        'time',
        '[class*="date"]',
        '[class*="published"]'
    ])
    for selector in date_selectors:
        element = soup.select_one(selector)
        if element:
            if selector == 'meta[property="article:published_time"]':
                metadata['date'] = element.get('content', '').strip()
            else:
                metadata['date'] = element.text.strip()
            if metadata['date']:
                break
    
    # Extract category
    category_selectors = site_selectors.get('category', [
        '.category',
        '[class*="category"]',
        '.breadcrumb',
        '[class*="breadcrumb"]'
    ])
    for selector in category_selectors:
        element = soup.select_one(selector)
        if element:
            metadata['category'] = element.text.strip()
            if metadata['category']:
                break
    
    # Determine content type
    url_lower = url.lower()
    for content_type, keywords in CONTENT_TYPES.items():
        if any(keyword in url_lower for keyword in keywords):
            metadata['content_type'] = content_type
            break
    
    return metadata

def extract_content(soup: BeautifulSoup, url: str) -> str:
    """Extract content using site-specific selectors."""
    site_selectors = get_site_selectors(url)
    content_selectors = site_selectors.get('content', [
        'article', 
        '.article-content', 
        '.story-content', 
        '#article-body',
        'div[class*="article"]',
        'div[class*="content"]',
        'div[class*="story"]',
        'main',
        'div[role="main"]'
    ])
    
    article_text = ""
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
            if len(text) > len(article_text):
                article_text = text
    
    if not article_text:
        paragraphs = soup.find_all('p')
        if paragraphs:
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
    
    return article_text

def determine_content_type(url: str, content: str) -> str:
    """Determine the type of content based on URL and content."""
    url_lower = url.lower()
    content_lower = content.lower()
    
    # Check URL patterns first
    for content_type, keywords in CONTENT_TYPES.items():
        if any(keyword in url_lower for keyword in keywords):
            return content_type
    
    # Check content patterns if URL doesn't match
    if any(keyword in content_lower for keyword in ['news', 'report', 'update']):
        return 'news'
    elif any(keyword in content_lower for keyword in ['analysis', 'research', 'study']):
        return 'analysis'
    elif any(keyword in content_lower for keyword in ['definition', 'term', 'glossary']):
        return 'definition'
    elif any(keyword in content_lower for keyword in ['tutorial', 'guide', 'how to']):
        return 'tutorial'
    elif any(keyword in content_lower for keyword in ['market', 'trading', 'investing']):
        return 'market'
    
    return 'unknown'

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT_PERIOD)
def extract_article_content(url: str, max_retries: int = 3) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """Extract article content using Playwright with stealth mode and retry logic."""
    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
                page = context.new_page()
                stealth_sync(page)
                
                time.sleep(random.uniform(1, 3))
                
                response = page.goto(url, timeout=30000, wait_until="domcontentloaded")
                if not response:
                    logging.error(f"Failed to load page: {url}")
                    continue
                    
                if response.status != 200:
                    logging.error(f"HTTP {response.status} for {url}")
                    continue
                
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except:
                    pass
                
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract metadata
                metadata = extract_metadata(soup, url)
                
                # Extract title
                title = metadata['title']
                if not title:
                    title = "No title found"
                
                # Extract content
                article_text = extract_content(soup, url)
                
                browser.close()
                
                if not article_text:
                    logging.warning(f"No content found for {url}")
                    continue
                
                # Clean and format content
                article_text = clean_text(article_text)
                
                # Update metadata
                metadata['word_count'] = len(article_text.split())
                metadata['content_type'] = determine_content_type(url, article_text)
                
                return title, article_text, metadata
                
        except Exception as e:
            logging.error(f"Error extracting content from {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(2, 5))
                continue
            return None, None, None
    
    return None, None, None

def scrape(url, template, topic_embedding):
    try:
        print(f"\n[+] Scraping: {url}")
        
        # Handle PDF files
        if is_pdf_url(url):
            print("[i] Processing PDF file")
            content = extract_pdf_text(url)
            if not content:
                return None
            return {
                "url": url,
                "title": os.path.basename(url).replace('.pdf', ''),
                "content": content[:2000]
            }
        
        # Choose between Playwright and requests based on the site
        if is_javascript_site(url):
            print("[i] Using Playwright for JavaScript-rendered content")
            html = fetch_dynamic_html(url)
            if not html:
                return None
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            res = requests.get(url, timeout=10, headers=headers)
            res.raise_for_status()
            html = res.text
        
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title = extract_field(soup, template["fields"]["title"]["selectors"])
        if not title:
            print(f"[-] No title found for {url}")
            return None
        print(f"[✓] Title: {title[:100]}...")

        # Extract content
        content = extract_field(soup, template["fields"]["content"]["selectors"])
        if not content:
            print(f"[-] No content found for {url}")
            print("\n[DEBUG] First 1000 chars of HTML:")
            print(soup.prettify()[:1000])
            return None

        if len(content) < template["fields"]["content"].get("filters", {}).get("min_length", 100):
            print(f"[-] Skipping {url}: content too short ({len(content)} chars)")
            return None

        print(f"[✓] Content length: {len(content)} chars")

        # Calculate similarity (temporarily disabled for debugging)
        content_embedding = model.encode([content])
        sim = np.dot(topic_embedding, content_embedding[0]) / (
            np.linalg.norm(topic_embedding) * np.linalg.norm(content_embedding[0])
        )
        print(f"[=] Similarity score: {sim:.2f}")

        # Temporarily disable similarity filtering for debugging
        return {"url": url, "title": title, "content": content[:2000]}

    except requests.exceptions.RequestException as e:
        print(f"[!] Network error for {url}: {e}")
        return None
    except Exception as e:
        print(f"[!] Failed to scrape {url}: {e}")
        return None

def save_embeddings(data, output_dir="data", domain="finance"):
    """Save embeddings and texts with proper formatting for RAG system."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare texts and embeddings
        texts = []
        embeddings = []
        
        for item in data:
            if "text" in item and "embedding" in item:
                texts.append({
                    "text": item["text"],
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "metadata": item.get("metadata", {})
                })
                embeddings.append(item["embedding"])
        
        if not texts or not embeddings:
            raise ValueError("No valid data to save")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create and save FAISS index
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array.astype('float32'))
        faiss.write_index(index, f"{output_dir}/{domain}_index.faiss")
        
        # Save texts
        with open(f"{output_dir}/{domain}_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Successfully saved {len(texts)} documents to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error saving embeddings: {str(e)}")
        raise

def is_url_valid(url):
    """Check if URL is valid and accessible using GET request with proper headers"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        response = requests.get(
            url, 
            headers=headers, 
            allow_redirects=True, 
            timeout=10
        )
        if response.status_code == 200:
            return True
        else:
            logging.warning(f"URL returned status {response.status_code}: {url}")
            return False
    except requests.exceptions.Timeout:
        logging.warning(f"Timeout while checking URL: {url}")
        return False
    except requests.exceptions.ConnectionError:
        logging.warning(f"Connection error while checking URL: {url}")
        return False
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to check URL {url} - Reason: {e}")
        return False

def is_finance_relevant(text, keywords):
    """Check if text contains enough finance-related keywords"""
    text_lower = text.lower()
    count = sum(1 for kw in keywords if kw.lower() in text_lower)
    return count >= 2

def get_matched_keywords(text, keywords):
    """Get list of matched keywords in text"""
    return [kw for kw in keywords if kw.lower() in text.lower()]

def process_articles():
    """Process articles and create embeddings."""
    try:
        # Create output directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('articles', exist_ok=True)
        
        # Load URLs
        with open('urls.txt', 'r') as f:
            urls = [line.strip() for line in f 
                    if line.strip() and not line.strip().startswith('#')]
        
        processed_data = []
        
        # Process each URL
        for url in urls:
            try:
                logging.info(f"Processing: {url}")
                
                # Validate URL
                if not is_url_valid(url):
                    logging.warning(f"Invalid URL: {url} - Reason: URL check failed")
                    continue
                
                # Extract content and metadata
                title, content, metadata = extract_article_content(url)
                if not content:
                    logging.warning(f"No content extracted from: {url}")
                    continue
                
                # Check if content is finance-related
                if not is_finance_relevant(content, FINANCE_KEYWORDS):
                    logging.info(f"Content not finance-related: {url}")
                    continue
                
                # Get matched keywords
                matched_keywords = get_matched_keywords(content, FINANCE_KEYWORDS)
                
                # Create embedding
                embedding = model.encode(content).tolist()
                
                # Prepare article data
                article_data = {
                    "url": url,
                    "title": title,
                    "text": content,
                    "matched_keywords": matched_keywords,
                    "embedding": embedding,
                    "metadata": metadata
                }
                
                # Save individual article
                url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
                filename = f"articles/{url_hash}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(article_data, f, ensure_ascii=False, indent=2)
                
                processed_data.append(article_data)
                logging.info(f"Successfully processed: {url}")
                
                # Random delay between requests
                time.sleep(random.uniform(3, 7))
                
            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")
                continue
        
        # Save all embeddings and texts
        if processed_data:
            save_embeddings(processed_data)
            logging.info(f"Successfully processed {len(processed_data)} articles")
        else:
            logging.warning("No articles were successfully processed")
            
    except Exception as e:
        logging.error(f"Error in process_articles: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        process_articles()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1) 