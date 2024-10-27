import os
import sys
import time
import logging
import threading
import queue
import json
import random
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import List, Dict, Optional
from collections import deque

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configure logging
logger = logging.getLogger('news_scraper')
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Constants
FMP_API_KEY = os.getenv('FMP_API_KEY')
FMP_BASE_URL = 'https://financialmodelingprep.com/api/v3/stock_news'
REQUESTS_PER_MINUTE = 10
MINUTE_WINDOW = 60
REQUEST_SPACING = MINUTE_WINDOW / REQUESTS_PER_MINUTE

# OpenAI Constants
OPENAI_MODEL = "gpt-4"
MAX_TOKENS = 4096
TEMPERATURE = 0.5

# User agent list for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
]

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

class ScrapingStatus:
    def __init__(self):
        self.total_urls = 0
        self.processed_urls = 0
        self.success_count = 0
        self.error_count = 0
        self.ai_processed_count = 0
        self.ai_error_count = 0
        self.is_complete = False
        self.start_time = None
        self.end_time = None

    def start(self, total_urls):
        self.total_urls = total_urls
        self.start_time = datetime.now()

    def update(self, success: bool):
        self.processed_urls += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def update_ai_status(self, success: bool):
        if success:
            self.ai_processed_count += 1
        else:
            self.ai_error_count += 1

    def complete(self):
        self.is_complete = True
        self.end_time = datetime.now()

    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return timedelta(0)

    @property
    def progress(self):
        return (self.processed_urls / self.total_urls * 100) if self.total_urls > 0 else 0

class ScrapingData:
    def __init__(self):
        self.articles = deque(maxlen=1000)
        self.status = ScrapingStatus()
        self.lock = threading.Lock()
        self.subscribers = set()

scraping_data = ScrapingData()
message_queue = queue.Queue()

def add_message(message: str, category: str = 'info'):
    message_queue.put((message, category))

def get_messages():
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get_nowait())
    return messages

def get_random_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

def enhance_content_with_ai(content: str, title: str, symbol: str) -> Dict[str, str]:
    """Enhance content using OpenAI API with simplified response structure"""
    try:
        system_message = """You are a financial news analyst. Analyze the provided article and return your analysis in the following JSON format EXACTLY (maintain the structure with no deviations):
{
    "summary": "2-3 sentence executive summary here",
    "key_points": ["point 1", "point 2", "point 3"]
}

Be precise and ensure the response is valid JSON. Do not include any text outside the JSON structure."""

        prompt = f"""
Stock: {symbol}
Title: {title}
Content: {content}

Analyze this content and provide a concise analysis strictly following the JSON format specified in the system message."""

        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=MAX_TOKENS
            )

            response_text = response.choices[0].message.content.strip()
            response_text = response_text.replace('```json', '').replace('```', '')
            
            try:
                enhanced_content = json.loads(response_text)
                logger.info(f"Successfully enhanced content for {symbol}: {title}")
                
                if not all(key in enhanced_content for key in ['summary', 'key_points']):
                    raise KeyError("Missing required keys in response")
                
                if not isinstance(enhanced_content['key_points'], list):
                    enhanced_content['key_points'] = [enhanced_content['key_points']]
                
                return enhanced_content

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {str(e)}")
                logger.error(f"Raw response: {response_text[:500]}...")
                
                sections = response_text.split('\n\n')
                summary = next((s for s in sections if 'summary' in s.lower()), '')[:200]
                key_points = [p.strip('- ') for p in response_text.split('\n') if p.strip().startswith('-')][:4]
                
                return {
                    "summary": summary or "Error parsing summary",
                    "key_points": key_points or ["Content available but parsing failed"]
                }

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in AI content enhancement: {str(e)}")
        return {
            "summary": "Error generating summary",
            "key_points": ["Error processing content"]
        }

def scrape_url(url: str) -> Dict:
    try:
        if is_video_site(url):
            return {
                "success": True,
                "data": {
                    "extract": "[Video content - please visit the original URL]",
                    "sourceURL": url
                }
            }

        logger.info(f"Scraping URL: {url}")
        time.sleep(random.uniform(2, 5))

        response = requests.get(url, headers=get_random_headers(), timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Clean the HTML
        for element in soup.find_all(['script', 'style', 'iframe', 'nav', 'footer']):
            element.decompose()

        # Find main content
        content = None
        for tag in [
            soup.find('article'),
            soup.find(class_=lambda x: x and 'article' in x.lower()),
            soup.find(class_=lambda x: x and 'content' in x.lower()),
            soup.find('main'),
        ]:
            if tag:
                content = tag
                break

        if not content:
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = ' '.join(p.get_text().strip() for p in paragraphs)

        if not content:
            content = soup.get_text(separator='\n\n', strip=True)

        cleaned_content = content.get_text(separator='\n\n').strip() if not isinstance(content, str) else content
        cleaned_content = ' '.join(cleaned_content.split())

        return {
            "success": True,
            "data": {
                "extract": cleaned_content or "[No content could be extracted]",
                "sourceURL": url
            }
        }

    except Exception as e:
        logger.error(f"Scraping error for {url}: {str(e)}")
        return {
            "success": False,
            "data": {
                "extract": f"[Error scraping content: {str(e)}]",
                "sourceURL": url
            }
        }

def format_article_data(news_item, scraped_data):
    content = scraped_data.get('data', {}).get('extract', 'No content available') if scraped_data else 'Failed to scrape'
    title = news_item.get('title', 'No title')
    symbol = news_item.get('symbol', 'Unknown')

    try:
        enhanced_data = enhance_content_with_ai(content, title, symbol)
        scraping_data.status.update_ai_status(True)
    except Exception as e:
        logger.error(f"AI enhancement failed: {str(e)}")
        enhanced_data = {
            "summary": "Content processing failed",
            "key_points": ["Unable to process article content"]
        }
        scraping_data.status.update_ai_status(False)

    return {
        'title': title,
        'published_date': news_item.get('publishedDate', 'Unknown date'),
        'symbol': symbol,
        'url': news_item.get('url', ''),
        'summary': enhanced_data['summary'],
        'key_points': enhanced_data['key_points'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': urlparse(news_item.get('url', '')).netloc
    }

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([
            result.scheme in ('http', 'https'),
            result.netloc,
            '.' in result.netloc,
            len(result.netloc.split('.')) >= 2,
            result.netloc.split('.')[-1].lower() not in ['local', 'internal', 'test']
        ])
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
        return False

def is_blocked_source(url: str) -> bool:
    # Updated to only allow the most reliable sources through FMP API
    allowed_domains = [

        'businesswire.com', 
        'globenewswire.com',
        'prnewswire.com',
        'reuters.com',
    ]
    parsed_url = urlparse(url)
    return not any(domain in parsed_url.netloc.lower() for domain in allowed_domains)

def is_video_site(url: str) -> bool:
    video_domains = [
        'youtube.com', 'youtu.be', 'vimeo.com',
        'dailymotion.com', 'twitch.tv', 'tiktok.com',
        'facebook.com/watch', 'rumble.com'
    ]
    parsed_url = urlparse(url)
    return any(domain in parsed_url.netloc.lower() for domain in video_domains)

def fetch_news_data(api_key: str, base_url: str, tickers: List[str], limit: int = 100) -> List[Dict]:
    tickers_str = ','.join(tickers)
    valid_articles = []
    page = 0
    total_attempts = 0
    max_attempts = 5  # Maximum number of API calls to prevent infinite loops
    
    while len(valid_articles) < limit and total_attempts < max_attempts:
        params = {
            'tickers': tickers_str,
            'limit': 50,  # Request more articles per batch
            'page': page,
            'apikey': api_key
        }

        try:
            logger.info(f"Fetching news page {page} for tickers: {tickers_str}")
            add_message(f"Fetching news batch {page + 1}")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            # Filter valid articles
            for article in data:
                url = article.get('url', '')
                if is_valid_url(url) and not is_blocked_source(url):
                    valid_articles.append(article)
                    if len(valid_articles) >= limit:
                        break
            
            page += 1
            total_attempts += 1
            
            if len(data) < 50:  # No more articles available
                break
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {str(e)}")
            add_message(f"Error fetching news data: {str(e)}", 'danger')
            break
        
        time.sleep(1)  # Respect API rate limits
    
    logger.info(f"Found {len(valid_articles)} valid articles after checking {total_attempts} pages")
    return valid_articles[:limit]

def notify_subscribers(data):
    with scraping_data.lock:
        dead_subscribers = set()
        for subscriber in scraping_data.subscribers:
            try:
                subscriber.put(json.dumps(data))
            except Exception as e:
                logger.error(f"Error notifying subscriber: {str(e)}")
                dead_subscribers.add(subscriber)
        scraping_data.subscribers.difference_update(dead_subscribers)

def run_scraper(tickers: List[str], limit: int):
    scraping_data.status = ScrapingStatus()
    scraping_data.status.start_time = datetime.now()

    if not FMP_API_KEY or not os.getenv('OPENAI_API_KEY'):
        error_msg = "Missing required API keys. Please check your environment variables."
        logger.error(error_msg)
        add_message(error_msg, 'danger')
        scraping_data.status.complete()
        return

    news_data = fetch_news_data(FMP_API_KEY, FMP_BASE_URL, tickers, limit)
    if not news_data:
        error_msg = "No news data to process"
        logger.error(error_msg)
        add_message(error_msg, 'danger')
        scraping_data.status.complete()
        return

    if len(news_data) < limit:
        add_message(f"Only found {len(news_data)} valid articles from allowed sources", 'warning')

    total_urls = len(news_data)
    logger.info(f"Processing {total_urls} valid articles")
    scraping_data.status.start(total_urls)

    try:
        for i, news_item in enumerate(news_data):
            url = news_item.get('url', '')
            scraped_data = scrape_url(url)
            
            try:
                article_data = format_article_data(news_item, scraped_data)
                notify_subscribers(article_data)
                scraping_data.status.update(True)
            except Exception as e:
                logger.error(f"Processing error for {url}: {str(e)}")
                scraping_data.status.update(False)

            # Update status with AI processing metrics
            status_data = {
                'progress': scraping_data.status.progress,
                'processed': scraping_data.status.processed_urls,
                'total': scraping_data.status.total_urls,
                'success_count': scraping_data.status.success_count,
                'error_count': scraping_data.status.error_count,
                'ai_processed': scraping_data.status.ai_processed_count,
                'ai_errors': scraping_data.status.ai_error_count
            }
            notify_subscribers(status_data)

            # Rate limiting
            if i < total_urls - 1:
                time.sleep(random.uniform(2, 5))

    except Exception as e:
        logger.error(f"An error occurred during scraping: {str(e)}")
        add_message(str(e), 'danger')

    finally:
        scraping_data.status.complete()
        notify_subscribers({'status': 'complete'})

# Flask Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    """Handle scraping request"""
    tickers = request.form.get('tickers', '')
    limit = request.form.get('limit', '10')

    if not tickers:
        flash("Please provide at least one ticker symbol.", 'danger')
        return redirect(url_for('index'))

    try:
        tickers_list = [ticker.strip().upper() for ticker in tickers.split(',') if ticker.strip()]
        limit = int(limit)
        if limit < 1 or limit > 100:
            flash("Limit must be between 1 and 100.", 'danger')
            return redirect(url_for('index'))
    except ValueError:
        flash("Invalid input for limit.", 'danger')
        return redirect(url_for('index'))

    # Clear old data and start new scraping thread
    with scraping_data.lock:
        scraping_data.articles.clear()

    thread = threading.Thread(target=run_scraper, args=(tickers_list, limit))
    thread.daemon = True
    thread.start()

    flash("Scraping started! Please wait for it to complete.", 'info')
    return redirect(url_for('results'))

@app.route('/results')
def results():
    """Display scraping results"""
    for message, category in get_messages():
        flash(message, category)

    with scraping_data.lock:
        articles = list(scraping_data.articles)
    return render_template('results.html',
                         articles=articles,
                         status=scraping_data.status)

@app.route('/stream')
def stream():
    """SSE endpoint for real-time updates"""
    def event_stream():
        subscriber_queue = queue.Queue()
        
        with scraping_data.lock:
            scraping_data.subscribers.add(subscriber_queue)
        
        try:
            while True:
                try:
                    data = subscriber_queue.get(timeout=30)  # 30 second timeout
                    if isinstance(data, str):
                        yield f"data: {data}\n\n"
                    elif isinstance(data, dict):
                        yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'keepalive': True})}\n\n"
                except Exception as e:
                    logger.error(f"Error in event stream: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except GeneratorExit:
            with scraping_data.lock:
                scraping_data.subscribers.remove(subscriber_queue)

    return Response(event_stream(),
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Transfer-Encoding': 'chunked',
                       'Connection': 'keep-alive'
                   })

@app.route('/status')
def get_status():
    """API endpoint for current scraping status"""
    return {
        'progress': scraping_data.status.progress,
        'processed': scraping_data.status.processed_urls,
        'total': scraping_data.status.total_urls,
        'success_count': scraping_data.status.success_count,
        'error_count': scraping_data.status.error_count,
        'ai_processed': scraping_data.status.ai_processed_count,
        'ai_errors': scraping_data.status.ai_error_count,
        'is_complete': scraping_data.status.is_complete,
        'duration': str(scraping_data.status.duration)
    }

@app.route('/export')
def export_data():
    """Export scraped data as JSON"""
    with scraping_data.lock:
        articles = list(scraping_data.articles)

    response = Response(
        json.dumps(articles, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment;filename=news_data.json'}
    )
    return response

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'api_keys': {
            'fmp': bool(FMP_API_KEY),
            'openai': bool(os.getenv('OPENAI_API_KEY'))
        }
    }

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500

def validate_environment():
    """Validate required environment variables"""
    required_vars = ['FMP_API_KEY', 'OPENAI_API_KEY', 'FLASK_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        return False
    return True

if __name__ == "__main__":
    if not validate_environment():
        sys.exit(1)

    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting application on port {port}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )