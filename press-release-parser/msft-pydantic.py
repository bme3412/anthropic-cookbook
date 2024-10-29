from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
import json
from datetime import datetime
import logging
from typing import Optional, List
import re
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SegmentResults(BaseModel):
    """Model for onsemi's business segment results"""
    name: str
    revenue: float
    sequential_change: Optional[float]
    year_over_year_change: Optional[float]

class FinancialMetrics(BaseModel):
    """Model for onsemi's key financial metrics"""
    revenue: float = Field(description="Revenue in millions of dollars")
    gaap_gross_margin: float = Field(description="GAAP gross margin percentage")
    non_gaap_gross_margin: float = Field(description="Non-GAAP gross margin percentage")
    gaap_operating_margin: float = Field(description="GAAP operating margin percentage")
    non_gaap_operating_margin: float = Field(description="Non-GAAP operating margin percentage")
    gaap_eps: float = Field(description="GAAP diluted earnings per share")
    non_gaap_eps: float = Field(description="Non-GAAP diluted earnings per share")

class GuidanceMetrics(BaseModel):
    """Model for onsemi's guidance metrics"""
    revenue: str
    gross_margin: str
    operating_expenses: str
    diluted_shares: str

class EarningsData(BaseModel):
    """Model for onsemi's earnings release structure"""
    fiscal_quarter: str
    fiscal_year: str
    report_date: str
    financial_metrics: FinancialMetrics
    segment_results: List[SegmentResults]
    guidance: GuidanceMetrics
    highlights: List[str]
    ceo_quote: Optional[str]

def extract_number(text: str) -> Optional[float]:
    """Extract a number from text, handling both percentages and dollar amounts"""
    try:
        # Remove commas and convert to float
        number = re.search(r'[\d,]+\.?\d*', text)
        if number:
            return float(number.group().replace(',', ''))
        return None
    except:
        return None

def extract_earnings_data(content: str) -> EarningsData:
    """Extract structured earnings data from the raw content"""
    
    # Extract fiscal period
    quarter_match = re.search(r'Reports (\w+) Quarter (\d{4})', content)
    fiscal_quarter = quarter_match.group(1) if quarter_match else ""
    fiscal_year = quarter_match.group(2) if quarter_match else ""
    
    # Extract report date
    date_match = re.search(r'[A-Z][a-z]+ \d{1,2}, \d{4}', content)
    report_date = date_match.group(0) if date_match else ""
    
    # Extract revenue
    revenue_match = re.search(r'Revenue of \$?([\d,]+\.?\d*)', content)
    revenue = extract_number(revenue_match.group(1)) if revenue_match else 0.0
    
    # Extract margins
    gaap_gm_match = re.search(r'GAAP gross margin.*?(\d+\.?\d*)%', content)
    non_gaap_gm_match = re.search(r'non-GAAP gross margin.*?(\d+\.?\d*)%', content)
    gaap_om_match = re.search(r'GAAP operating margin.*?(\d+\.?\d*)%', content)
    non_gaap_om_match = re.search(r'non-GAAP operating margin.*?(\d+\.?\d*)%', content)
    
    # Extract EPS
    gaap_eps_match = re.search(r'GAAP diluted earnings per share.*?\$(\d+\.?\d*)', content)
    non_gaap_eps_match = re.search(r'non-GAAP diluted earnings per share.*?\$(\d+\.?\d*)', content)
    
    financial_metrics = FinancialMetrics(
        revenue=revenue,
        gaap_gross_margin=float(gaap_gm_match.group(1)) if gaap_gm_match else 0.0,
        non_gaap_gross_margin=float(non_gaap_gm_match.group(1)) if non_gaap_gm_match else 0.0,
        gaap_operating_margin=float(gaap_om_match.group(1)) if gaap_om_match else 0.0,
        non_gaap_operating_margin=float(non_gaap_om_match.group(1)) if non_gaap_om_match else 0.0,
        gaap_eps=float(gaap_eps_match.group(1)) if gaap_eps_match else 0.0,
        non_gaap_eps=float(non_gaap_eps_match.group(1)) if non_gaap_eps_match else 0.0
    )
    
    # Extract segment results
    segment_results = []
    segment_pattern = r'(?P<name>PSG|AMG|ISG)\s+\$\s*(?P<revenue>[\d,]+\.?\d*)\s+.*?\((?P<seq_change>\d+)%\)\s+\((?P<yoy_change>\d+)%\)'
    for match in re.finditer(segment_pattern, content):
        segment_results.append(SegmentResults(
            name=match.group('name'),
            revenue=extract_number(match.group('revenue')),
            sequential_change=float(match.group('seq_change')),
            year_over_year_change=float(match.group('yoy_change'))
        ))
    
    # Extract guidance
    guidance_section = re.search(r'FOURTH QUARTER.*?OUTLOOK.*?\$(.*?)(?=\*|\n\n)', content, re.DOTALL)
    if guidance_section:
        guidance_text = guidance_section.group(1)
        
        revenue_guidance = re.search(r'\$?([\d,]+) to \$?([\d,]+)\s*million', guidance_text)
        gross_margin_guidance = re.search(r'(\d+\.?\d*)% to (\d+\.?\d*)%', guidance_text)
        opex_guidance = re.search(r'\$(\d+) to \$(\d+)\s*million', guidance_text)
        shares_guidance = re.search(r'(\d+)\s*million', guidance_text)
        
        guidance = GuidanceMetrics(
            revenue=f"${revenue_guidance.group(1)} to ${revenue_guidance.group(2)} million" if revenue_guidance else "",
            gross_margin=f"{gross_margin_guidance.group(1)}% to {gross_margin_guidance.group(2)}%" if gross_margin_guidance else "",
            operating_expenses=f"${opex_guidance.group(1)} to ${opex_guidance.group(2)} million" if opex_guidance else "",
            diluted_shares=f"{shares_guidance.group(1)} million" if shares_guidance else ""
        )
    else:
        guidance = GuidanceMetrics(revenue="", gross_margin="", operating_expenses="", diluted_shares="")
    
    # Extract highlights
    highlights = []
    highlights_section = re.search(r'highlights:(.*?)(?="|\n\n)', content, re.DOTALL | re.IGNORECASE)
    if highlights_section:
        highlights = [h.strip() for h in highlights_section.group(1).split('\n') if h.strip()]
    
    # Extract CEO quote
    ceo_quote_match = re.search(r'"([^"]*El-Khoury[^"]*)"', content)
    ceo_quote = ceo_quote_match.group(0) if ceo_quote_match else None
    
    return EarningsData(
        fiscal_quarter=fiscal_quarter,
        fiscal_year=fiscal_year,
        report_date=report_date,
        financial_metrics=financial_metrics,
        segment_results=segment_results,
        guidance=guidance,
        highlights=highlights,
        ceo_quote=ceo_quote
    )

def scrape_onsemi_earnings(api_key: str, url: str) -> Optional[dict]:
    """Scrape onsemi earnings data using Firecrawl"""
    try:
        logger.info(f"Starting scrape of URL: {url}")
        
        app = FirecrawlApp(api_key=api_key)
        result = app.scrape_url(
            url,
            params={
                'formats': ['markdown'],
                'onlyMainContent': True
            }
        )

        if not result or 'markdown' not in result:
            logger.error("No markdown content returned from scraper")
            return None

        earnings_data = extract_earnings_data(result['markdown'])
        
        processed_data = {
            "metadata": {
                "scrape_date": datetime.now().isoformat(),
                "source_url": url,
                "scrape_status": "success",
            },
            "extracted_data": earnings_data.model_dump(),
            "raw_markdown": result['markdown']
        }

        logger.info("Successfully scraped and processed earnings data")
        return processed_data

    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        return {
            "metadata": {
                "scrape_date": datetime.now().isoformat(),
                "source_url": url,
                "scrape_status": "error",
                "error_message": str(e)
            }
        }

def print_earnings_summary(data: dict) -> None:
    """Print a formatted summary of the earnings data"""
    if not data or "extracted_data" not in data:
        logger.warning("No extracted data available to summarize")
        return

    extracted = data["extracted_data"]
    
    print("\n=== onsemi Earnings Summary ===")
    print(f"Quarter: {extracted['fiscal_quarter']} {extracted['fiscal_year']}")
    print(f"Report Date: {extracted['report_date']}")
    
    print("\nFinancial Metrics:")
    metrics = extracted['financial_metrics']
    print(f"Revenue: ${metrics['revenue']}M")
    print(f"GAAP Gross Margin: {metrics['gaap_gross_margin']}%")
    print(f"Non-GAAP Gross Margin: {metrics['non_gaap_gross_margin']}%")
    print(f"GAAP Operating Margin: {metrics['gaap_operating_margin']}%")
    print(f"Non-GAAP Operating Margin: {metrics['non_gaap_operating_margin']}%")
    print(f"GAAP EPS: ${metrics['gaap_eps']}")
    print(f"Non-GAAP EPS: ${metrics['non_gaap_eps']}")
    
    print("\nSegment Results:")
    for segment in extracted['segment_results']:
        print(f"\n{segment['name']}:")
        print(f"Revenue: ${segment['revenue']}M")
        print(f"Sequential Change: {segment['sequential_change']}%")
        print(f"Year-over-Year Change: {segment['year_over_year_change']}%")
    
    print("\nGuidance:")
    guidance = extracted['guidance']
    print(f"Revenue: {guidance['revenue']}")
    print(f"Gross Margin: {guidance['gross_margin']}")
    print(f"Operating Expenses: {guidance['operating_expenses']}")
    print(f"Diluted Shares: {guidance['diluted_shares']}")
    
    if extracted.get('highlights'):
        print("\nKey Highlights:")
        for idx, highlight in enumerate(extracted['highlights'], 1):
            print(f"{idx}. {highlight}")
    
    if extracted.get('ceo_quote'):
        print(f"\nCEO Quote:\n{extracted['ceo_quote']}")

if __name__ == "__main__":
    # Replace with your actual Firecrawl API key
    FIRECRAWL_API_KEY = "#"
    
    url = "https://investor.onsemi.com/news-releases/news-release-details/onsemi-reports-third-quarter-2024-results"
    logger.info(f"Using URL: {url}")
    
    earnings_data = scrape_onsemi_earnings(FIRECRAWL_API_KEY, url)
    
    if earnings_data:
        filename = f"onsemi_earnings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(earnings_data, f, indent=2)
            logger.info(f"Data saved to {filename}")
        
        print_earnings_summary(earnings_data)
    else:
        logger.error("Failed to scrape earnings data")