import os
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
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, Response

# Load environment variables and configure logging
load_dotenv()

# Configure logging to file and console
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

# User agent list for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
]

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Global variables and classes for real-time updates


class ScrapingStatus:
    def __init__(self):
        self.total_urls = 0
        self.processed_urls = 0
        self.success_count = 0
        self.error_count = 0
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
        self.articles = deque(maxlen=1000)  # Store last 1000 articles
        self.status = ScrapingStatus()
        self.lock = threading.Lock()
        self.subscribers = set()


scraping_data = ScrapingData()
message_queue = queue.Queue()


def add_message(message: str, category: str = 'info'):
    """Thread-safe way to add messages that will be displayed to the user."""
    message_queue.put((message, category))


def get_messages():
    """Retrieve all messages from the queue."""
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get_nowait())
    return messages


def get_random_headers():
    """Generate random headers for web requests"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


def is_valid_url(url: str) -> bool:
    """Enhanced URL validation with additional checks."""
    try:
        result = urlparse(url)
        if not all([
            result.scheme in ('http', 'https'),
            result.netloc,
            '.' in result.netloc
        ]):
            return False

        domain_parts = result.netloc.split('.')
        if len(domain_parts) < 2:
            return False

        tld = domain_parts[-1].lower()
        if len(tld) < 2 or tld in ['local', 'internal', 'test']:
            return False

        return True
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
        return False


def fetch_news_data(api_key: str, base_url: str, tickers: List[str], limit: int = 100) -> List[Dict]:
    """Fetch news data from FMP API with error handling."""
    tickers_str = ','.join(tickers)
    params = {
        'tickers': tickers_str,
        'limit': limit,
        'apikey': api_key
    }

    try:
        logger.info(f"Fetching news for tickers: {tickers_str}")
        add_message(f"Fetching news for tickers: {tickers_str}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} news articles")
        add_message(f"Successfully fetched {len(data)} news articles")
        return data
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching news data: {str(e)}"
        logger.error(error_msg)
        add_message(error_msg, 'danger')
        return []


def is_video_site(url: str) -> bool:
    """Check if the URL is from a video platform that shouldn't be scraped."""
    video_domains = [
        'youtube.com', 'youtu.be',
        'vimeo.com',
        'dailymotion.com',
        'twitch.tv',
        'tiktok.com',
        'facebook.com/watch',
        'rumble.com'
    ]
    parsed_url = urlparse(url)
    return any(domain in parsed_url.netloc.lower() for domain in video_domains)


def scrape_url(url: str) -> Optional[Dict]:
    """Scrape article content directly from the website, skipping video platforms"""
    try:
        # Skip video platforms
        if is_video_site(url):
            logger.info(f"Skipping video platform URL: {url}")
            return {
                "success": True,
                "data": {
                    "extract": "[This is a video content. Please visit the original URL to view the video.]",
                    "sourceURL": url
                }
            }

        logger.info(f"Scraping URL: {url}")

        # Add delay to avoid overwhelming servers
        time.sleep(random.uniform(2, 5))

        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'iframe', 'nav', 'footer']):
            element.decompose()

        # Extract main content based on common article containers
        content = None
        possible_content_tags = [
            soup.find('article'),
            soup.find(class_=lambda x: x and 'article' in x.lower()),
            soup.find(class_=lambda x: x and 'content' in x.lower()),
            soup.find(class_=lambda x: x and 'story' in x.lower()),
            soup.find('main'),
        ]

        for tag in possible_content_tags:
            if tag:
                content = tag
                break

        if not content:
            # Fallback to looking for clusters of paragraphs
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = ' '.join(p.get_text().strip() for p in paragraphs)

        if not content:
            logger.warning(f"No content found for URL: {url}")
            return None

        # Clean the extracted content
        if isinstance(content, str):
            cleaned_content = content
        else:
            cleaned_content = content.get_text(separator='\n\n').strip()

        # Remove extra whitespace and normalize spacing
        cleaned_content = ' '.join(cleaned_content.split())

        return {
            "success": True,
            "data": {
                "extract": cleaned_content,
                "sourceURL": url
            }
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for URL {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Scraping error for URL {url}: {str(e)}")
        return None


def handle_rate_limit(batch_number: int, total_batches: int):
    """Handle rate limiting between requests."""
    if batch_number < total_batches:
        wait_time = random.uniform(3, 7)  # Random delay between requests
        logger.info(f"Rate limit pause: {wait_time:.1f} seconds")
        time.sleep(wait_time)


def format_article_data(news_item, scraped_data):
    return {
        'title': news_item.get('title', 'No title'),
        'published_date': news_item.get('publishedDate', 'Unknown date'),
        'symbol': news_item.get('symbol', 'Unknown'),
        'url': news_item.get('url', ''),
        'content': scraped_data.get('data', {}).get('extract', 'No content available') if scraped_data else 'Failed to scrape',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def notify_subscribers(article_data):
    """Notify all subscribers of new article data"""
    with scraping_data.lock:
        dead_subscribers = set()
        for subscriber in scraping_data.subscribers:
            try:
                subscriber.put(article_data)
            except:
                dead_subscribers.add(subscriber)
        # Remove dead subscribers
        scraping_data.subscribers.difference_update(dead_subscribers)


def is_blocked_source(url: str) -> bool:
    """Check if the URL is from a blocked source."""
    blocked_domains = [
        'fool.com',
        'seekingalpha.com',
        'investors.com',
        'barrons.com',
        'wsj.com',
        'bloomberg.com',
        'ft.com',
        'youtube.com',
        'youtu.be',
        'twitter.com',
        'x.com',
        'facebook.com',
        'instagram.com',
        'tiktok.com',
        'reddit.com',
        'linkedin.com'
    ]
    parsed_url = urlparse(url)
    return any(domain in parsed_url.netloc.lower() for domain in blocked_domains)


def is_valid_content(scraped_data: Optional[Dict]) -> bool:
    """Check if the scraped content is valid and substantial."""
    if not scraped_data or not scraped_data.get('data', {}).get('extract'):
        return False

    content = scraped_data['data']['extract']
    # Check if content is substantial (at least 100 characters)
    if len(content) < 100:
        return False

    return True


def run_scraper(tickers: List[str], limit: int):
    """Run the scraping process with source filtering"""
    scraping_data.status.start_time = datetime.now()

    if not FMP_API_KEY:
        error_msg = "Missing FMP API key. Please check your environment variables."
        logger.error(error_msg)
        add_message(error_msg, 'danger')
        scraping_data.status.complete()
        return

    # Fetch news data
    news_data = fetch_news_data(FMP_API_KEY, FMP_BASE_URL, tickers, limit)
    if not news_data:
        error_msg = "No news data to process"
        logger.error(error_msg)
        add_message(error_msg, 'danger')
        scraping_data.status.complete()
        return

    df_news = pd.DataFrame(news_data)

    # Filter out blocked sources first
    news_links = df_news['url'].tolist()
    valid_news_links = [
        url for url in news_links
        if is_valid_url(url) and not is_blocked_source(url)
    ]

    if not valid_news_links:
        error_msg = "No valid news sources found after filtering"
        logger.error(error_msg)
        add_message(error_msg, 'danger')
        scraping_data.status.complete()
        return

    total_urls = len(valid_news_links)
    logger.info(f"Found {total_urls} valid articles after filtering sources")
    scraping_data.status.start(total_urls)

    try:
        for i, url in enumerate(valid_news_links):
            news_item = df_news[df_news['url'] == url].iloc[0].to_dict()
            scraped_data = scrape_url(url)

            # Only process and send valid content
            if is_valid_content(scraped_data):
                article_data = format_article_data(news_item, scraped_data)

                with scraping_data.lock:
                    scraping_data.articles.append(article_data)

                # Notify subscribers of new article
                notify_subscribers(article_data)
                scraping_data.status.update(True)
            else:
                scraping_data.status.update(False)

            # Send status update
            status_data = {
                'progress': scraping_data.status.progress,
                'processed': scraping_data.status.processed_urls,
                'total': scraping_data.status.total_urls,
                'success_count': scraping_data.status.success_count,
                'error_count': scraping_data.status.error_count
            }
            notify_subscribers(status_data)

            handle_rate_limit(i + 1, total_urls)

    except Exception as e:
        logger.error(f"An error occurred during scraping: {str(e)}")
        add_message(str(e), 'danger')

    finally:
        scraping_data.status.complete()
        notify_subscribers({'status': 'complete'})

# Flask routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scrape', methods=['POST'])
def scrape():
    tickers = request.form.get('tickers', '')
    limit = request.form.get('limit', '10')

    if not tickers:
        flash("Please provide at least one ticker symbol.", 'danger')
        return redirect(url_for('index'))

    try:
        tickers_list = [ticker.strip().upper()
                        for ticker in tickers.split(',') if ticker.strip()]
        limit = int(limit)
        if limit < 1 or limit > 100:
            flash("Limit must be between 1 and 100.", 'danger')
            return redirect(url_for('index'))
    except ValueError:
        flash("Invalid input for limit.", 'danger')
        return redirect(url_for('index'))

    # Reset status and clear old messages
    scraping_data.status = ScrapingStatus()
    while not message_queue.empty():
        message_queue.get()

    # Start scraping in a separate thread
    thread = threading.Thread(target=run_scraper, args=(tickers_list, limit))
    thread.daemon = True
    thread.start()

    flash("Scraping started! Please wait for it to complete.", 'info')
    return redirect(url_for('results'))


@app.route('/results')
def results():
    for message, category in get_messages():
        flash(message, category)

    with scraping_data.lock:
        articles = list(scraping_data.articles)
    return render_template('results.html', articles=articles, status=scraping_data.status)


@app.route('/stream')
def stream():
    """SSE endpoint for real-time updates"""
    def event_stream():
        subscriber_queue = queue.Queue()
        with scraping_data.lock:
            scraping_data.subscribers.add(subscriber_queue)

        try:
            while True:
                data = subscriber_queue.get()
                if isinstance(data, dict):
                    yield f"data: {json.dumps(data)}\n\n"
        except GeneratorExit:
            with scraping_data.lock:
                scraping_data.subscribers.remove(subscriber_queue)

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/status')
def get_status():
    """API endpoint to get current scraping status."""
    return {
        'progress': scraping_data.status.progress,
        'processed': scraping_data.status.processed_urls,
        'total': scraping_data.status.total_urls,
        'success_count': scraping_data.status.success_count,
        'error_count': scraping_data.status.error_count,
        'is_complete': scraping_data.status.is_complete,
        'duration': str(scraping_data.status.duration)
    }


if __name__ == "__main__":
    app.run(debug=True)
