import os
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Optional
import json

@dataclass
class FinancialMetrics:
    revenue: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    earnings_per_share: Optional[float] = None
    segment_revenue: Dict[str, Optional[float]] = None
    cloud_metrics: Dict[str, Optional[float]] = None
    cash_flow_metrics: Dict[str, Optional[float]] = None
    balance_sheet_metrics: Dict[str, Optional[float]] = None

    def __post_init__(self):
        # Initialize empty dictionaries if None
        if self.segment_revenue is None:
            self.segment_revenue = {}
        if self.cloud_metrics is None:
            self.cloud_metrics = {}
        if self.cash_flow_metrics is None:
            self.cash_flow_metrics = {}
        if self.balance_sheet_metrics is None:
            self.balance_sheet_metrics = {}

def read_text_file(file_path: str) -> str:
    """Read content from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error reading .txt file: {str(e)}")

def create_extraction_prompt(text: str) -> str:
    return f"""You are a financial analyst assistant. Analyze this financial report and extract the metrics in JSON format.

Metrics to extract (all monetary values in millions USD):
1. Key metrics:
   - Revenue (total revenue)
   - Operating income
   - Net income
   - Diluted earnings per share

2. Segment revenue:
   - Productivity and Business Processes
   - Intelligent Cloud
   - More Personal Computing

3. Cloud and product metrics:
   - Microsoft Cloud Revenue (total)
   - Azure Growth (percentage)
   - Microsoft Cloud Revenue growth y/y %
   - Microsoft Cloud Revenue growth constant currency y/y %
   - Microsoft 365 Commercial products and cloud services revenue growth y/y %
   - Microsoft 365 Commercial products and cloud services revenue growth constant currency y/y %
   - Microsoft 365 Commercial cloud revenue growth y/y %
   - Microsoft 365 Commercial cloud revenue growth constant currency y/y %
   - Microsoft 365 Consumer products and cloud services revenue growth y/y %
   - Microsoft 365 Consumer products and cloud services revenue growth constant currency y/y %
   - Microsoft 365 Consumer cloud revenue growth y/y %
   - Microsoft 365 Consumer cloud revenue growth constant currency y/y %
   - LinkedIn revenue growth y/y %
   - LinkedIn revenue growth constant currency y/y %
   - Dynamics products and cloud services revenue growth y/y %
   - Dynamics products and cloud services revenue growth constant currency y/y %

4. Cash flow metrics:
   - Operating Cash Flow (Net cash from operations)
   - Capital Expenditure (Additions to property and equipment)

5. Balance sheet metrics:
   - Cash and Investments (Total cash, cash equivalents, and short-term investments)
   - Total Assets
   - Total Liabilities
   - Stockholders Equity (Total stockholders' equity)

Return a JSON object with these exact keys and numeric values only:
{{
    "revenue": 0,
    "operating_income": 0,
    "net_income": 0,
    "earnings_per_share": 0,
    "segment_revenue": {{
        "Productivity and Business Processes": 0,
        "Intelligent Cloud": 0,
        "More Personal Computing": 0
    }},
    "cloud_metrics": {{
        "Microsoft Cloud Revenue": 0,
        "Azure Growth": 0,
        "Microsoft Cloud Growth": 0,
        "Microsoft Cloud Growth Constant Currency": 0,
        "M365 Commercial Growth": 0,
        "M365 Commercial Growth Constant Currency": 0,
        "M365 Commercial Cloud Growth": 0,
        "M365 Commercial Cloud Growth Constant Currency": 0,
        "M365 Consumer Growth": 0,
        "M365 Consumer Growth Constant Currency": 0,
        "M365 Consumer Cloud Growth": 0,
        "M365 Consumer Cloud Growth Constant Currency": 0,
        "LinkedIn Growth": 0,
        "LinkedIn Growth Constant Currency": 0,
        "Dynamics Growth": 0,
        "Dynamics Growth Constant Currency": 0
    }},
    "cash_flow_metrics": {{
        "Operating Cash Flow": 0,
        "Capital Expenditure": 0
    }},
    "balance_sheet_metrics": {{
        "Cash and Investments": 0,
        "Total Assets": 0,
        "Total Liabilities": 0,
        "Stockholders Equity": 0
    }}
}}

Extract all growth percentages as numbers (e.g., 15 for 15% growth). Replace the 0 values with the actual numbers from the report. Use null for any missing values.

Financial report to analyze:
{text}"""

def parse_financial_report(api_key: str, text: str) -> FinancialMetrics:
    """Parse the financial report using OpenAI's API."""
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4 as fallback if gpt-4o isn't available
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant that extracts numerical data from financial reports. You only return JSON data, no explanations."},
                {"role": "user", "content": create_extraction_prompt(text)}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        metrics_dict = json.loads(response.choices[0].message.content)
        return FinancialMetrics(**metrics_dict)
    
    except Exception as e:
        raise Exception(f"Error parsing financial report with OpenAI: {str(e)}")

def format_metrics(metrics: FinancialMetrics) -> str:
    """Format the extracted metrics into a readable string with enhanced organization."""
    output = []
    output.append("Microsoft Financial Results Summary")
    output.append("=" * 50)
    
    def format_value(value: Optional[float], prefix: str = "$", suffix: str = "B", divisor: float = 1) -> str:
        if value is None:
            return "N/A"
        return f"{prefix}{value/divisor:.1f}{suffix}"
    
    def format_growth(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:+.1f}%"
    
    # Key Financial Metrics
    output.append("\nKey Financial Metrics")
    output.append("-" * 30)
    output.append(f"Revenue:           {format_value(metrics.revenue)}")
    output.append(f"Operating Income:  {format_value(metrics.operating_income)}")
    output.append(f"Net Income:        {format_value(metrics.net_income)}")
    output.append(f"Earnings Per Share: {format_value(metrics.earnings_per_share, prefix='$', suffix='')}")
    
    # Segment Revenue
    output.append("\nSegment Revenue")
    output.append("-" * 30)
    for segment, value in metrics.segment_revenue.items():
        # Add padding to align values
        output.append(f"{segment:<30} {format_value(value)}")
    
    # Cloud Services Performance
    output.append("\nCloud Services Performance")
    output.append("-" * 30)
    cloud_metrics = metrics.cloud_metrics
    output.append(f"Microsoft Cloud Revenue:    {format_value(cloud_metrics.get('Microsoft Cloud Revenue'))}")
    output.append(f"├─ Growth (y/y):           {format_growth(cloud_metrics.get('Microsoft Cloud Growth'))}")
    output.append(f"└─ Growth (constant currency): {format_growth(cloud_metrics.get('Microsoft Cloud Growth Constant Currency'))}")
    
    # Azure Performance
    output.append(f"\nAzure Growth (y/y):         {format_growth(cloud_metrics.get('Azure Growth'))}")
    
    # Microsoft 365 Commercial
    output.append("\nMicrosoft 365 Commercial Performance")
    output.append("-" * 30)
    output.append(f"Products and Cloud Services:")
    output.append(f"├─ Growth (y/y):           {format_growth(cloud_metrics.get('M365 Commercial Growth'))}")
    output.append(f"└─ Growth (constant currency): {format_growth(cloud_metrics.get('M365 Commercial Growth Constant Currency'))}")
    output.append(f"\nCloud Revenue:")
    output.append(f"├─ Growth (y/y):           {format_growth(cloud_metrics.get('M365 Commercial Cloud Growth'))}")
    output.append(f"└─ Growth (constant currency): {format_growth(cloud_metrics.get('M365 Commercial Cloud Growth Constant Currency'))}")
    
    # Microsoft 365 Consumer
    output.append("\nMicrosoft 365 Consumer Performance")
    output.append("-" * 30)
    output.append(f"Products and Cloud Services:")
    output.append(f"├─ Growth (y/y):           {format_growth(cloud_metrics.get('M365 Consumer Growth'))}")
    output.append(f"└─ Growth (constant currency): {format_growth(cloud_metrics.get('M365 Consumer Growth Constant Currency'))}")
    output.append(f"\nCloud Revenue:")
    output.append(f"├─ Growth (y/y):           {format_growth(cloud_metrics.get('M365 Consumer Cloud Growth'))}")
    output.append(f"└─ Growth (constant currency): {format_growth(cloud_metrics.get('M365 Consumer Cloud Growth Constant Currency'))}")
    
    # LinkedIn and Dynamics
    output.append("\nOther Product Lines")
    output.append("-" * 30)
    output.append("LinkedIn Revenue:")
    output.append(f"├─ Growth (y/y):           {format_growth(cloud_metrics.get('LinkedIn Growth'))}")
    output.append(f"└─ Growth (constant currency): {format_growth(cloud_metrics.get('LinkedIn Growth Constant Currency'))}")
    output.append("\nDynamics Products and Cloud Services:")
    output.append(f"├─ Growth (y/y):           {format_growth(cloud_metrics.get('Dynamics Growth'))}")
    output.append(f"└─ Growth (constant currency): {format_growth(cloud_metrics.get('Dynamics Growth Constant Currency'))}")
    
    # Cash Flow and Balance Sheet
    output.append("\nCash Flow and Balance Sheet (in billions)")
    output.append("-" * 30)
    output.append("Cash Flow:")
    for metric, value in metrics.cash_flow_metrics.items():
        output.append(f"├─ {metric:<25} {format_value(value, divisor=1000)}")
    
    output.append("\nBalance Sheet:")
    items = list(metrics.balance_sheet_metrics.items())
    for i, (metric, value) in enumerate(items):
        prefix = "└─ " if i == len(items) - 1 else "├─ "
        output.append(f"{prefix}{metric:<25} {format_value(value, divisor=1000)}")
    
    return "\n".join(output)

def main():
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Read the financial report from .txt file
    file_path = 'MSFT_FY25Q1.txt'
    try:
        text = read_text_file(file_path)
        metrics = parse_financial_report(api_key, text)
        
        # Print formatted results
        print(format_metrics(metrics))
        
        # Save to JSON
        with open('financial_metrics.json', 'w') as f:
            json.dump(vars(metrics), f, indent=2, default=lambda x: None)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()