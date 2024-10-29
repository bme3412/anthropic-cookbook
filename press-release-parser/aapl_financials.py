# aapl_financials.py

import pdfplumber
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict
import json
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


# Define Pydantic Models for Financial Statements

class IncomeStatement(BaseModel):
    net_sales_products: float
    net_sales_services: float
    total_net_sales: float
    cost_of_sales_products: float
    cost_of_sales_services: float
    total_cost_of_sales: float
    gross_margin: float
    operating_expenses_research_and_development: float
    operating_expenses_selling_general_administrative: float
    total_operating_expenses: float
    operating_income: float
    other_income_expense_net: float
    income_before_provision_for_income_taxes: float
    provision_for_income_taxes: float
    net_income: float
    earnings_per_share_basic: float
    earnings_per_share_diluted: float
    shares_used_basic: int
    shares_used_diluted: int
    net_sales_by_reportable_segment: Dict[str, float]
    net_sales_by_category: Dict[str, float]


class BalanceSheet(BaseModel):
    cash_and_cash_equivalents: float
    marketable_securities_current: float
    accounts_receivable_net: float
    vendor_non_trade_receivables: float
    inventories: float
    other_current_assets: float
    total_current_assets: float
    marketable_securities_non_current: float
    property_plant_equipment_net: float
    other_non_current_assets: float
    total_non_current_assets: float
    total_assets: float
    accounts_payable: float
    other_current_liabilities: float
    deferred_revenue: float
    commercial_paper: float
    term_debt_current: float
    total_current_liabilities: float
    term_debt_non_current: float
    other_non_current_liabilities: float
    total_non_current_liabilities: float
    total_liabilities: float
    shareholders_equity_common_stock: float
    shareholders_equity_accumulated_deficit: float
    shareholders_equity_accumulated_other_comprehensive_loss: float
    total_shareholders_equity: float
    total_liabilities_and_shareholders_equity: float


class CashFlowStatement(BaseModel):
    cash_beginning: float
    net_income: float
    depreciation_amortization: float
    share_based_compensation: float
    other_adjustments: float
    changes_in_operating_assets_and_liabilities: Dict[str, float]
    cash_generated_operating_activities: float
    investing_activities_purchases_marketable_securities: float
    investing_activities_proceeds_maturities_marketable_securities: float
    investing_activities_proceeds_sales_marketable_securities: float
    investing_activities_payments_pp_e: float
    investing_activities_other: float
    cash_generated_investing_activities: float
    financing_activities_payments_taxes_net_share_settlement: float
    financing_activities_payments_dividends: float
    financing_activities_repurchases_common_stock: float
    financing_activities_proceeds_issuance_term_debt: Optional[float]
    financing_activities_repayments_term_debt: float
    financing_activities_repayments_commercial_paper: float
    financing_activities_other: float
    cash_used_in_financing_activities: float
    increase_decrease_cash: float
    cash_ending: float
    supplemental_cash_flow_disclosure: Dict[str, float]


class AppleFinancials(BaseModel):
    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow_statement: CashFlowStatement


# Helper Functions

def make_unique_columns(columns: List[str]) -> List[str]:
    """
    Appends a suffix to duplicate column names to make them unique.

    Args:
        columns (List[str]): Original list of column names.

    Returns:
        List[str]: List of unique column names.
    """
    seen = {}
    unique_columns = []
    for col in columns:
        col_clean = col.strip()
        if col_clean in seen:
            seen[col_clean] += 1
            unique_columns.append(f"{col_clean}_{seen[col_clean]}")
        else:
            seen[col_clean] = 1
            unique_columns.append(col_clean)
    return unique_columns


def extract_first_number(value: str) -> str:
    """
    Extracts the first numerical value from a string, handling negative numbers in parentheses.

    Args:
        value (str): The string containing numerical values.

    Returns:
        str: The first numerical value found, negative if in parentheses, or '0' if none.
    """
    # Pattern to match numbers with optional parentheses for negatives and optional dollar signs
    match = re.search(r'\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?', value)
    if match:
        num_str = match.group(0)
        # Check if the number is negative (enclosed in parentheses)
        if num_str.startswith('(') and num_str.endswith(')'):
            num_str = '-' + num_str[1:-1]
        # Remove dollar signs and commas
        num_str = num_str.replace('$', '').replace(',', '')
        return num_str
    else:
        return '0'


def safe_float(value: str) -> Optional[float]:
    """
    Safely converts a string to a float. Returns None if conversion fails.

    Args:
        value (str): The string to convert.

    Returns:
        Optional[float]: The converted float or None.
    """
    try:
        return float(value)
    except (ValueError, AttributeError) as e:
        logging.warning(f"Could not convert value '{value}' to float: {e}")
        return None


def safe_int(value: str) -> Optional[int]:
    """
    Safely converts a string to an integer. Returns None if conversion fails.

    Args:
        value (str): The string to convert.

    Returns:
        Optional[int]: The converted integer or None.
    """
    try:
        return int(float(value))
    except (ValueError, AttributeError) as e:
        logging.warning(f"Could not convert value '{value}' to int: {e}")
        return None


def clean_key(key: str) -> str:
    """
    Cleans and formats the key string.

    Args:
        key (str): The key string to clean.

    Returns:
        str: The cleaned key string.
    """
    key = key.lower().strip()
    key = re.sub(r'[ /\.,()-]', '_', key)
    key = re.sub(r'_+', '_', key)  # Replace multiple underscores with single
    key = key.rstrip('_')  # Remove trailing underscores
    return key


def identify_and_concatenate_table(tables: List[pd.DataFrame], keywords: List[str]) -> Optional[pd.DataFrame]:
    """
    Identifies and concatenates tables that match specific keywords.

    Args:
        tables (List[pd.DataFrame]): List of extracted tables.
        keywords (List[str]): Keywords to search for.

    Returns:
        Optional[pd.DataFrame]: The concatenated table or None if not found.
    """
    matched_tables = []
    for table in tables:
        # Convert all cells to lowercase strings for case-insensitive search
        table_str = table.astype(str).apply(lambda x: ' '.join(x), axis=1).str.lower().str.cat(sep=' ')
        if any(keyword.lower() in table_str for keyword in keywords):
            matched_tables.append(table)
            logging.debug(f"Matched table with keywords {keywords}:\n{table}\n")

    if not matched_tables:
        return None

    # Concatenate all matched tables
    try:
        concatenated_table = pd.concat(matched_tables, ignore_index=True)
        logging.debug(f"Concatenated Table:\n{concatenated_table}\n")
        return concatenated_table
    except pd.errors.InvalidIndexError as e:
        logging.error(f"Pandas InvalidIndexError during concatenation: {e}")
        return None


# Function to Extract Tables from PDF using pdfplumber

def extract_tables_from_pdf(pdf_path: str) -> List[pd.DataFrame]:
    """
    Extracts tables from a PDF file using pdfplumber.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[pd.DataFrame]: A list of DataFrames extracted from the PDF.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            extracted_tables = page.extract_tables()
            logging.info(f"Page {page_number}: {len(extracted_tables)} tables found.")
            for table_index, table in enumerate(extracted_tables, start=1):
                if not table or len(table) < 2:
                    logging.warning(f"Page {page_number} Table {table_index} is empty or too small.")
                    continue
                # Make column names unique
                unique_cols = make_unique_columns(table[0])
                df = pd.DataFrame(table[1:], columns=unique_cols)
                tables.append(df)
                logging.debug(f"Page {page_number} Table {table_index}:\n{df}\n")
    logging.info(f"Total tables extracted: {len(tables)}")
    return tables


# Function to Save Extracted Tables as CSV for Inspection

def save_tables_as_csv(tables: List[pd.DataFrame], output_dir: str = "extracted_tables"):
    """
    Saves each extracted table as a separate CSV file for manual inspection.

    Args:
        tables (List[pd.DataFrame]): List of extracted tables.
        output_dir (str, optional): Directory to save CSV files. Defaults to "extracted_tables".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, table in enumerate(tables, start=1):
        file_path = os.path.join(output_dir, f"table_{idx}.csv")
        table.to_csv(file_path, index=False)
        logging.info(f"Saved Table {idx} to {file_path}")


# Function to Clean and Process Income Statement

def process_income_statement(tables: List[pd.DataFrame]) -> IncomeStatement:
    """
    Processes the Income Statement table and returns an IncomeStatement model.

    Args:
        tables (List[pd.DataFrame]): List of extracted tables.

    Returns:
        IncomeStatement: Parsed income statement data.
    """
    logging.info("Processing Income Statement...")
    # Define refined keywords based on manual inspection
    income_keywords = [
        'condensed consolidated statements of operations',
        'net sales',
        'revenue',
        'income statement',
        'total revenue'
    ]
    
    # Use the enhanced identify_and_concatenate_table function
    income_table = identify_and_concatenate_table(tables, income_keywords)
    
    if income_table is None:
        raise ValueError("Income Statement table not found.")
    
    logging.debug(f"Income Statement Table:\n{income_table}\n")
    
    # Initialize dictionaries for data
    income_data = {}
    net_sales_segment = {}
    net_sales_category = {}
    
    # Iterate through rows to extract data
    for index, row in income_table.iterrows():
        # Combine all elements of the row into a single string to handle merged cells
        row_combined = ' '.join([str(cell) for cell in row])
        key_match = re.search(r'(.*?):?\s*(.*)', row_combined)
        if key_match:
            key = clean_key(key_match.group(1))
            value_str = key_match.group(2).strip()
            
            # Skip section headers or empty values
            if key.endswith('activities') or key.endswith('assets') or key.endswith('liabilities') or key.endswith('equity') or key.endswith('segment') or key.endswith('category'):
                continue
            
            # Extract only the first numerical value
            value_extracted = extract_first_number(value_str)
            value = safe_float(value_extracted) or 0.0

            # Logging the extracted key and value
            logging.debug(f"Extracted Income Statement - Key: {key}, Value: {value}")
    
            # Assign values based on key patterns
            if 'net_sales_products' in key:
                income_data['net_sales_products'] = value
            elif 'net_sales_services' in key:
                income_data['net_sales_services'] = value
            elif 'total_net_sales' in key:
                income_data['total_net_sales'] = value
            elif 'cost_of_sales_products' in key:
                income_data['cost_of_sales_products'] = value
            elif 'cost_of_sales_services' in key:
                income_data['cost_of_sales_services'] = value
            elif 'total_cost_of_sales' in key:
                income_data['total_cost_of_sales'] = value
            elif 'gross_margin' in key:
                income_data['gross_margin'] = value
            elif 'operating_expenses_research_and_development' in key:
                income_data['operating_expenses_research_and_development'] = value
            elif 'operating_expenses_selling_general_administrative' in key:
                income_data['operating_expenses_selling_general_administrative'] = value
            elif 'total_operating_expenses' in key:
                income_data['total_operating_expenses'] = value
            elif 'operating_income' in key:
                income_data['operating_income'] = value
            elif 'other_income_expense_net' in key:
                income_data['other_income_expense_net'] = value
            elif 'income_before_provision_for_income_taxes' in key:
                income_data['income_before_provision_for_income_taxes'] = value
            elif 'provision_for_income_taxes' in key:
                income_data['provision_for_income_taxes'] = value
            elif 'net_income' in key:
                income_data['net_income'] = value
            elif 'earnings_per_share_basic' in key:
                income_data['earnings_per_share_basic'] = value
            elif 'earnings_per_share_diluted' in key:
                income_data['earnings_per_share_diluted'] = value
            elif 'shares_used_basic' in key:
                income_data['shares_used_basic'] = safe_int(value_extracted) or 0
            elif 'shares_used_diluted' in key:
                income_data['shares_used_diluted'] = safe_int(value_extracted) or 0
            elif 'net_sales_by_reportable_segment' in key:
                # Next rows contain segment data
                continue
            elif key in ['americas', 'europe', 'greater_china', 'japan', 'rest_of_asia_pacific']:
                net_sales_segment[key.replace('_', ' ').title()] = value
            elif 'net_sales_by_category' in key:
                # Next rows contain category data
                continue
            elif key in ['iphone', 'mac', 'ipad', 'wearables_home_accessories', 'services']:
                formatted_key = key.replace('wearables_home_accessories', 'Wearables, Home, Accessories').title()
                net_sales_category[formatted_key] = value

    # Populate the required fields
    try:
        income_statement = IncomeStatement(
            net_sales_products=income_data.get('net_sales_products', 0.0),
            net_sales_services=income_data.get('net_sales_services', 0.0),
            total_net_sales=income_data.get('total_net_sales', 0.0),
            cost_of_sales_products=income_data.get('cost_of_sales_products', 0.0),
            cost_of_sales_services=income_data.get('cost_of_sales_services', 0.0),
            total_cost_of_sales=income_data.get('total_cost_of_sales', 0.0),
            gross_margin=income_data.get('gross_margin', 0.0),
            operating_expenses_research_and_development=income_data.get('operating_expenses_research_and_development', 0.0),
            operating_expenses_selling_general_administrative=income_data.get('operating_expenses_selling_general_administrative', 0.0),
            total_operating_expenses=income_data.get('total_operating_expenses', 0.0),
            operating_income=income_data.get('operating_income', 0.0),
            other_income_expense_net=income_data.get('other_income_expense_net', 0.0),
            income_before_provision_for_income_taxes=income_data.get('income_before_provision_for_income_taxes', 0.0),
            provision_for_income_taxes=income_data.get('provision_for_income_taxes', 0.0),
            net_income=income_data.get('net_income', 0.0),
            earnings_per_share_basic=income_data.get('earnings_per_share_basic', 0.0),
            earnings_per_share_diluted=income_data.get('earnings_per_share_diluted', 0.0),
            shares_used_basic=income_data.get('shares_used_basic', 0),
            shares_used_diluted=income_data.get('shares_used_diluted', 0),
            net_sales_by_reportable_segment=net_sales_segment,
            net_sales_by_category=net_sales_category
        )
    except ValidationError as e:
        logging.error(f"Validation error in Income Statement: {e}")
        raise

    logging.info("Income Statement processed successfully.")
    return income_statement


# Function to Clean and Process Balance Sheet

def process_balance_sheet(tables: List[pd.DataFrame]) -> BalanceSheet:
    """
    Processes the Balance Sheet table and returns a BalanceSheet model.

    Args:
        tables (List[pd.DataFrame]): List of extracted tables.

    Returns:
        BalanceSheet: Parsed balance sheet data.
    """
    logging.info("Processing Balance Sheet...")
    # Define keywords that identify the Balance Sheet
    balance_keywords = ['condensed consolidated balance sheets', 'cash and cash equivalents']
    
    # Use the enhanced identify_and_concatenate_table function
    balance_table = identify_and_concatenate_table(tables, balance_keywords)
    
    if balance_table is None:
        raise ValueError("Balance Sheet table not found.")
    
    logging.debug(f"Balance Sheet Table:\n{balance_table}\n")
    
    # Initialize dictionary for data
    balance_data = {}
    
    # Iterate through rows to extract data
    for index, row in balance_table.iterrows():
        # Combine all elements of the row into a single string to handle merged cells
        row_combined = ' '.join([str(cell) for cell in row])
        key_match = re.search(r'(.*?):?\s*(.*)', row_combined)
        if key_match:
            key = clean_key(key_match.group(1))
            value_str = key_match.group(2).strip()
            
            # Skip section headers or empty values
            if key.endswith('activities') or key.endswith('assets') or key.endswith('liabilities') or key.endswith('equity'):
                continue
            
            # Extract only the first numerical value
            value_extracted = extract_first_number(value_str)
            value = safe_float(value_extracted) or 0.0

            # Logging the extracted key and value
            logging.debug(f"Extracted Balance Sheet - Key: {key}, Value: {value}")

            # Assign values based on key patterns
            if 'cash_and_cash_equivalents' in key:
                balance_data['cash_and_cash_equivalents'] = value
            elif 'marketable_securities_current' in key:
                balance_data['marketable_securities_current'] = value
            elif 'accounts_receivable_net' in key:
                balance_data['accounts_receivable_net'] = value
            elif 'vendor_non_trade_receivables' in key:
                balance_data['vendor_non_trade_receivables'] = value
            elif 'inventories' in key and 'inventory_turnover' not in key:
                balance_data['inventories'] = value
            elif 'other_current_assets' in key:
                balance_data['other_current_assets'] = value
            elif 'total_current_assets' in key:
                balance_data['total_current_assets'] = value
            elif 'marketable_securities_non_current' in key:
                balance_data['marketable_securities_non_current'] = value
            elif 'property_plant_equipment_net' in key:
                balance_data['property_plant_equipment_net'] = value
            elif 'other_non_current_assets' in key:
                balance_data['other_non_current_assets'] = value
            elif 'total_non_current_assets' in key:
                balance_data['total_non_current_assets'] = value
            elif 'total_assets' in key:
                balance_data['total_assets'] = value
            elif 'accounts_payable' in key and 'accounts_payable_turnover' not in key:
                balance_data['accounts_payable'] = value
            elif 'other_current_liabilities' in key:
                balance_data['other_current_liabilities'] = value
            elif 'deferred_revenue' in key:
                balance_data['deferred_revenue'] = value
            elif 'commercial_paper' in key:
                balance_data['commercial_paper'] = value
            elif 'term_debt_current' in key:
                balance_data['term_debt_current'] = value
            elif 'total_current_liabilities' in key:
                balance_data['total_current_liabilities'] = value
            elif 'term_debt_non_current' in key:
                balance_data['term_debt_non_current'] = value
            elif 'other_non_current_liabilities' in key:
                balance_data['other_non_current_liabilities'] = value
            elif 'total_non_current_liabilities' in key:
                balance_data['total_non_current_liabilities'] = value
            elif 'total_liabilities' in key and 'total_liabilities_and_shareholders_equity' not in key:
                balance_data['total_liabilities'] = value
            elif 'shareholders_equity_common_stock' in key:
                balance_data['shareholders_equity_common_stock'] = value
            elif 'shareholders_equity_accumulated_deficit' in key:
                balance_data['shareholders_equity_accumulated_deficit'] = value
            elif 'shareholders_equity_accumulated_other_comprehensive_loss' in key:
                balance_data['shareholders_equity_accumulated_other_comprehensive_loss'] = value
            elif 'total_shareholders_equity' in key:
                balance_data['total_shareholders_equity'] = value
            elif 'total_liabilities_and_shareholders_equity' in key:
                balance_data['total_liabilities_and_shareholders_equity'] = value

    # Populate the required fields
    try:
        balance_sheet = BalanceSheet(
            cash_and_cash_equivalents=balance_data.get('cash_and_cash_equivalents', 0.0),
            marketable_securities_current=balance_data.get('marketable_securities_current', 0.0),
            accounts_receivable_net=balance_data.get('accounts_receivable_net', 0.0),
            vendor_non_trade_receivables=balance_data.get('vendor_non_trade_receivables', 0.0),
            inventories=balance_data.get('inventories', 0.0),
            other_current_assets=balance_data.get('other_current_assets', 0.0),
            total_current_assets=balance_data.get('total_current_assets', 0.0),
            marketable_securities_non_current=balance_data.get('marketable_securities_non_current', 0.0),
            property_plant_equipment_net=balance_data.get('property_plant_equipment_net', 0.0),
            other_non_current_assets=balance_data.get('other_non_current_assets', 0.0),
            total_non_current_assets=balance_data.get('total_non_current_assets', 0.0),
            total_assets=balance_data.get('total_assets', 0.0),
            accounts_payable=balance_data.get('accounts_payable', 0.0),
            other_current_liabilities=balance_data.get('other_current_liabilities', 0.0),
            deferred_revenue=balance_data.get('deferred_revenue', 0.0),
            commercial_paper=balance_data.get('commercial_paper', 0.0),
            term_debt_current=balance_data.get('term_debt_current', 0.0),
            total_current_liabilities=balance_data.get('total_current_liabilities', 0.0),
            term_debt_non_current=balance_data.get('term_debt_non_current', 0.0),
            other_non_current_liabilities=balance_data.get('other_non_current_liabilities', 0.0),
            total_non_current_liabilities=balance_data.get('total_non_current_liabilities', 0.0),
            total_liabilities=balance_data.get('total_liabilities', 0.0),
            shareholders_equity_common_stock=balance_data.get('shareholders_equity_common_stock', 0.0),
            shareholders_equity_accumulated_deficit=balance_data.get('shareholders_equity_accumulated_deficit', 0.0),
            shareholders_equity_accumulated_other_comprehensive_loss=balance_data.get('shareholders_equity_accumulated_other_comprehensive_loss', 0.0),
            total_shareholders_equity=balance_data.get('total_shareholders_equity', 0.0),
            total_liabilities_and_shareholders_equity=balance_data.get('total_liabilities_and_shareholders_equity', 0.0)
        )
    except ValidationError as e:
        logging.error(f"Validation error in Balance Sheet: {e}")
        raise

    logging.info("Balance Sheet processed successfully.")
    return balance_sheet


# Function to Clean and Process Cash Flow Statement

def process_cash_flow_statement(tables: List[pd.DataFrame]) -> CashFlowStatement:
    """
    Processes the Cash Flow Statement table and returns a CashFlowStatement model.

    Args:
        tables (List[pd.DataFrame]): List of extracted tables.

    Returns:
        CashFlowStatement: Parsed cash flow statement data.
    """
    logging.info("Processing Cash Flow Statement...")
    # Define keywords that identify the Cash Flow Statement
    cash_flow_keywords = ['condensed consolidated statements of cash flows', 'cash, cash equivalents and restricted cash']
    
    # Use the enhanced identify_and_concatenate_table function
    cash_flow_table = identify_and_concatenate_table(tables, cash_flow_keywords)
    
    if cash_flow_table is None:
        raise ValueError("Cash Flow Statement table not found.")
    
    logging.debug(f"Cash Flow Statement Table:\n{cash_flow_table}\n")
    
    # Initialize dictionaries for data
    cash_flow_data = {}
    supplemental_disclosures = {}
    changes_operating = {}
    
    # Iterate through rows to extract data
    for index, row in cash_flow_table.iterrows():
        # Combine all elements of the row into a single string to handle merged cells
        row_combined = ' '.join([str(cell) for cell in row])
        key_match = re.search(r'(.*?):?\s*(.*)', row_combined)
        if key_match:
            key = clean_key(key_match.group(1))
            value_str = key_match.group(2).strip()
            
            # Skip section headers or empty values
            if key.endswith('activities') or key.endswith('assets') or key.endswith('liabilities') or key.endswith('equity') or key.endswith('disclosure'):
                continue
            
            # Extract only the first numerical value
            value_extracted = extract_first_number(value_str)
            value = safe_float(value_extracted) or 0.0

            # Logging the extracted key and value
            logging.debug(f"Extracted Cash Flow Statement - Key: {key}, Value: {value}")

            # Assign values based on key patterns
            if 'cash_beginning' in key:
                cash_flow_data['cash_beginning'] = value
            elif 'net_income' in key:
                cash_flow_data['net_income'] = value
            elif 'depreciation_amortization' in key:
                cash_flow_data['depreciation_amortization'] = value
            elif 'share_based_compensation' in key:
                cash_flow_data['share_based_compensation'] = value
            elif 'other_adjustments' in key:
                cash_flow_data['other_adjustments'] = value
            elif 'accounts_receivable_net' in key:
                changes_operating['accounts_receivable_net'] = value
            elif 'vendor_non_trade_receivables' in key:
                changes_operating['vendor_non_trade_receivables'] = value
            elif 'inventories' in key:
                changes_operating['inventories'] = value
            elif 'other_current_and_non_current_assets' in key:
                changes_operating['other_current_and_non_current_assets'] = value
            elif 'accounts_payable' in key:
                changes_operating['accounts_payable'] = value
            elif 'other_current_and_non_current_liabilities' in key:
                changes_operating['other_current_and_non_current_liabilities'] = value
            elif 'cash_generated_operating_activities' in key:
                cash_flow_data['cash_generated_operating_activities'] = value
            elif 'investing_activities_purchases_marketable_securities' in key:
                cash_flow_data['investing_activities_purchases_marketable_securities'] = value
            elif 'investing_activities_proceeds_maturities_marketable_securities' in key:
                cash_flow_data['investing_activities_proceeds_maturities_marketable_securities'] = value
            elif 'investing_activities_proceeds_sales_marketable_securities' in key:
                cash_flow_data['investing_activities_proceeds_sales_marketable_securities'] = value
            elif 'investing_activities_payments_pp_e' in key:
                cash_flow_data['investing_activities_payments_pp_e'] = value
            elif 'investing_activities_other' in key:
                cash_flow_data['investing_activities_other'] = value
            elif 'cash_generated_investing_activities' in key:
                cash_flow_data['cash_generated_investing_activities'] = value
            elif 'financing_activities_payments_taxes_net_share_settlement' in key:
                cash_flow_data['financing_activities_payments_taxes_net_share_settlement'] = value
            elif 'financing_activities_payments_dividends' in key:
                cash_flow_data['financing_activities_payments_dividends'] = value
            elif 'financing_activities_repurchases_common_stock' in key:
                cash_flow_data['financing_activities_repurchases_common_stock'] = value
            elif 'financing_activities_proceeds_issuance_term_debt' in key:
                cash_flow_data['financing_activities_proceeds_issuance_term_debt'] = value
            elif 'financing_activities_repayments_term_debt' in key:
                cash_flow_data['financing_activities_repayments_term_debt'] = value
            elif 'financing_activities_repayments_commercial_paper' in key:
                cash_flow_data['financing_activities_repayments_commercial_paper'] = value
            elif 'financing_activities_other' in key:
                cash_flow_data['financing_activities_other'] = value
            elif 'cash_used_in_financing_activities' in key:
                cash_flow_data['cash_used_in_financing_activities'] = value
            elif 'increase_decrease_cash' in key:
                cash_flow_data['increase_decrease_cash'] = value
            elif 'cash_ending' in key:
                cash_flow_data['cash_ending'] = value
            elif 'cash_paid_for_income_taxes_net' in key:
                supplemental_disclosures['cash_paid_for_income_taxes_net'] = value
            elif 'cash_paid_for_income_taxes_net_previous' in key:
                supplemental_disclosures['cash_paid_for_income_taxes_net_previous'] = value

    # Populate the required fields
    try:
        cash_flow_statement = CashFlowStatement(
            cash_beginning=cash_flow_data.get('cash_beginning', 0.0),
            net_income=cash_flow_data.get('net_income', 0.0),
            depreciation_amortization=cash_flow_data.get('depreciation_amortization', 0.0),
            share_based_compensation=cash_flow_data.get('share_based_compensation', 0.0),
            other_adjustments=cash_flow_data.get('other_adjustments', 0.0),
            changes_in_operating_assets_and_liabilities=changes_operating,
            cash_generated_operating_activities=cash_flow_data.get('cash_generated_operating_activities', 0.0),
            investing_activities_purchases_marketable_securities=cash_flow_data.get('investing_activities_purchases_marketable_securities', 0.0),
            investing_activities_proceeds_maturities_marketable_securities=cash_flow_data.get('investing_activities_proceeds_maturities_marketable_securities', 0.0),
            investing_activities_proceeds_sales_marketable_securities=cash_flow_data.get('investing_activities_proceeds_sales_marketable_securities', 0.0),
            investing_activities_payments_pp_e=cash_flow_data.get('investing_activities_payments_pp_e', 0.0),
            investing_activities_other=cash_flow_data.get('investing_activities_other', 0.0),
            cash_generated_investing_activities=cash_flow_data.get('cash_generated_investing_activities', 0.0),
            financing_activities_payments_taxes_net_share_settlement=cash_flow_data.get('financing_activities_payments_taxes_net_share_settlement', 0.0),
            financing_activities_payments_dividends=cash_flow_data.get('financing_activities_payments_dividends', 0.0),
            financing_activities_repurchases_common_stock=cash_flow_data.get('financing_activities_repurchases_common_stock', 0.0),
            financing_activities_proceeds_issuance_term_debt=cash_flow_data.get('financing_activities_proceeds_issuance_term_debt'),
            financing_activities_repayments_term_debt=cash_flow_data.get('financing_activities_repayments_term_debt', 0.0),
            financing_activities_repayments_commercial_paper=cash_flow_data.get('financing_activities_repayments_commercial_paper', 0.0),
            financing_activities_other=cash_flow_data.get('financing_activities_other', 0.0),
            cash_used_in_financing_activities=cash_flow_data.get('cash_used_in_financing_activities', 0.0),
            increase_decrease_cash=cash_flow_data.get('increase_decrease_cash', 0.0),
            cash_ending=cash_flow_data.get('cash_ending', 0.0),
            supplemental_cash_flow_disclosure=supplemental_disclosures
        )
    except ValidationError as e:
        logging.error(f"Validation error in Cash Flow Statement: {e}")
        raise

    logging.info("Cash Flow Statement processed successfully.")
    return cash_flow_statement


# Function to Process the Entire PDF and Create AppleFinancials Model

def process_pdf(pdf_path: str) -> AppleFinancials:
    """
    Processes the PDF file and returns an AppleFinancials model.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        AppleFinancials: Parsed financial data.
    """
    tables = extract_tables_from_pdf(pdf_path)

    if len(tables) < 3:
        logging.warning("Less than 3 tables extracted from the PDF.")

    income_statement = process_income_statement(tables)
    balance_sheet = process_balance_sheet(tables)
    cash_flow_statement = process_cash_flow_statement(tables)

    apple_financials = AppleFinancials(
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow_statement=cash_flow_statement
    )

    return apple_financials


# Function to Save Financials to JSON

def save_financials(financials: AppleFinancials, output_path: str):
    """
    Saves the financial data to a JSON file.

    Args:
        financials (AppleFinancials): The financial data to save.
        output_path (str): Path to the output JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(financials.dict(), f, indent=2, ensure_ascii=False)
    logging.info(f"Financial data saved to {output_path}")


# Function to Format and Print Results

def format_results(financials: AppleFinancials) -> None:
    """
    Prints formatted financial analysis results.

    Args:
        financials (AppleFinancials): The financial data to print.
    """
    print("\nApple Q3 Fiscal Year 2024 Financial Analysis")
    print("=" * 50)

    # Income Statement
    print("\nIncome Statement:")
    is_ = financials.income_statement
    print(f"• Net Sales - Products: ${is_.net_sales_products} million")
    print(f"• Net Sales - Services: ${is_.net_sales_services} million")
    print(f"• Total Net Sales: ${is_.total_net_sales} million")
    print(f"• Cost of Sales - Products: ${is_.cost_of_sales_products} million")
    print(f"• Cost of Sales - Services: ${is_.cost_of_sales_services} million")
    print(f"• Total Cost of Sales: ${is_.total_cost_of_sales} million")
    print(f"• Gross Margin: ${is_.gross_margin} million")
    print(f"• Operating Expenses - R&D: ${is_.operating_expenses_research_and_development} million")
    print(f"• Operating Expenses - SG&A: ${is_.operating_expenses_selling_general_administrative} million")
    print(f"• Total Operating Expenses: ${is_.total_operating_expenses} million")
    print(f"• Operating Income: ${is_.operating_income} million")
    print(f"• Other Income/(Expense), Net: ${is_.other_income_expense_net} million")
    print(f"• Income Before Provision for Income Taxes: ${is_.income_before_provision_for_income_taxes} million")
    print(f"• Provision for Income Taxes: ${is_.provision_for_income_taxes} million")
    print(f"• Net Income: ${is_.net_income} million")
    print(f"• Earnings Per Share (Basic): ${is_.earnings_per_share_basic}")
    print(f"• Earnings Per Share (Diluted): ${is_.earnings_per_share_diluted}")
    print(f"• Shares Used (Basic): {is_.shares_used_basic} thousand")
    print(f"• Shares Used (Diluted): {is_.shares_used_diluted} thousand")

    # Net Sales by Reportable Segment
    print("\n• Net Sales by Reportable Segment:")
    if is_.net_sales_by_reportable_segment:
        for segment, amount in is_.net_sales_by_reportable_segment.items():
            print(f"  - {segment}: ${amount} million")
    else:
        print("  - No data available.")

    # Net Sales by Category
    print("\n• Net Sales by Category:")
    if is_.net_sales_by_category:
        for category, amount in is_.net_sales_by_category.items():
            print(f"  - {category}: ${amount} million")
    else:
        print("  - No data available.")

    # Balance Sheet
    print("\nBalance Sheet:")
    bs = financials.balance_sheet
    print(f"• Cash and Cash Equivalents: ${bs.cash_and_cash_equivalents} million")
    print(f"• Marketable Securities (Current): ${bs.marketable_securities_current} million")
    print(f"• Accounts Receivable, Net: ${bs.accounts_receivable_net} million")
    print(f"• Vendor Non-trade Receivables: ${bs.vendor_non_trade_receivables} million")
    print(f"• Inventories: ${bs.inventories} million")
    print(f"• Other Current Assets: ${bs.other_current_assets} million")
    print(f"• Total Current Assets: ${bs.total_current_assets} million")
    print(f"• Marketable Securities (Non-Current): ${bs.marketable_securities_non_current} million")
    print(f"• Property, Plant, and Equipment, Net: ${bs.property_plant_equipment_net} million")
    print(f"• Other Non-current Assets: ${bs.other_non_current_assets} million")
    print(f"• Total Non-current Assets: ${bs.total_non_current_assets} million")
    print(f"• Total Assets: ${bs.total_assets} million")
    print(f"• Accounts Payable: ${bs.accounts_payable} million")
    print(f"• Other Current Liabilities: ${bs.other_current_liabilities} million")
    print(f"• Deferred Revenue: ${bs.deferred_revenue} million")
    print(f"• Commercial Paper: ${bs.commercial_paper} million")
    print(f"• Term Debt (Current): ${bs.term_debt_current} million")
    print(f"• Total Current Liabilities: ${bs.total_current_liabilities} million")
    print(f"• Term Debt (Non-Current): ${bs.term_debt_non_current} million")
    print(f"• Other Non-current Liabilities: ${bs.other_non_current_liabilities} million")
    print(f"• Total Non-current Liabilities: ${bs.total_non_current_liabilities} million")
    print(f"• Total Liabilities: ${bs.total_liabilities} million")
    print(f"• Shareholders’ Equity - Common Stock: ${bs.shareholders_equity_common_stock} million")
    print(f"• Shareholders’ Equity - Accumulated Deficit: ${bs.shareholders_equity_accumulated_deficit} million")
    print(f"• Shareholders’ Equity - Accumulated Other Comprehensive Loss: ${bs.shareholders_equity_accumulated_other_comprehensive_loss} million")
    print(f"• Total Shareholders’ Equity: ${bs.total_shareholders_equity} million")
    print(f"• Total Liabilities and Shareholders’ Equity: ${bs.total_liabilities_and_shareholders_equity} million")

    # Cash Flow Statement
    print("\nCash Flow Statement:")
    cfs = financials.cash_flow_statement
    print(f"• Cash, Cash Equivalents, and Restricted Cash (Beginning): ${cfs.cash_beginning} million")
    print(f"• Net Income: ${cfs.net_income} million")
    print(f"• Depreciation and Amortization: ${cfs.depreciation_amortization} million")
    print(f"• Share-based Compensation Expense: ${cfs.share_based_compensation} million")
    print(f"• Other Adjustments: ${cfs.other_adjustments} million")
    print(f"• Changes in Operating Assets and Liabilities:")
    if cfs.changes_in_operating_assets_and_liabilities:
        for change, amount in cfs.changes_in_operating_assets_and_liabilities.items():
            formatted_change = change.replace('_', ' ').capitalize()
            print(f"  - {formatted_change}: ${amount} million")
    else:
        print("  - No data available.")
    print(f"• Cash Generated by Operating Activities: ${cfs.cash_generated_operating_activities} million")
    print(f"• Investing Activities - Purchases of Marketable Securities: ${cfs.investing_activities_purchases_marketable_securities} million")
    print(f"• Investing Activities - Proceeds from Maturities of Marketable Securities: ${cfs.investing_activities_proceeds_maturities_marketable_securities} million")
    print(f"• Investing Activities - Proceeds from Sales of Marketable Securities: ${cfs.investing_activities_proceeds_sales_marketable_securities} million")
    print(f"• Investing Activities - Payments for Acquisition of Property, Plant and Equipment: ${cfs.investing_activities_payments_pp_e} million")
    print(f"• Investing Activities - Other: ${cfs.investing_activities_other} million")
    print(f"• Cash Generated by Investing Activities: ${cfs.cash_generated_investing_activities} million")
    print(f"• Financing Activities - Payments for Taxes Related to Net Share Settlement of Equity Awards: ${cfs.financing_activities_payments_taxes_net_share_settlement} million")
    print(f"• Financing Activities - Payments for Dividends and Dividend Equivalents: ${cfs.financing_activities_payments_dividends} million")
    print(f"• Financing Activities - Repurchases of Common Stock: ${cfs.financing_activities_repurchases_common_stock} million")
    print(f"• Financing Activities - Proceeds from Issuance of Term Debt: ${cfs.financing_activities_proceeds_issuance_term_debt} million")
    print(f"• Financing Activities - Repayments of Term Debt: ${cfs.financing_activities_repayments_term_debt} million")
    print(f"• Financing Activities - Repayments of Commercial Paper: ${cfs.financing_activities_repayments_commercial_paper} million")
    print(f"• Financing Activities - Other: ${cfs.financing_activities_other} million")
    print(f"• Cash Used in Financing Activities: ${cfs.cash_used_in_financing_activities} million")
    print(f"• Increase/(Decrease) in Cash: ${cfs.increase_decrease_cash} million")
    print(f"• Cash, Cash Equivalents, and Restricted Cash (Ending): ${cfs.cash_ending} million")
    print(f"• Supplemental Cash Flow Disclosure:")
    if cfs.supplemental_cash_flow_disclosure:
        for disclosure, amount in cfs.supplemental_cash_flow_disclosure.items():
            formatted_disclosure = disclosure.replace('_', ' ').capitalize()
            print(f"  - {formatted_disclosure}: ${amount} million")
    else:
        print("  - No data available.")

    print("\nFull financial data has been saved to the JSON file.")


# Main Function

def main():
    """
    Main function to process the PDF and save the financial data.
    """
    PDF_PATH = "AAPL_Q3_2024.pdf"
    OUTPUT_PATH = "apple_financials_analysis.json"

    if not os.path.exists(PDF_PATH):
        logging.error(f"PDF file not found: {PDF_PATH}")
        return

    try:
        logging.info("Extracting tables from PDF...")
        tables = extract_tables_from_pdf(PDF_PATH)
        
        # Save tables as CSV for manual inspection
        save_tables_as_csv(tables)

        logging.info("Processing Income Statement...")
        income_statement = process_income_statement(tables)

        logging.info("Processing Balance Sheet...")
        balance_sheet = process_balance_sheet(tables)

        logging.info("Processing Cash Flow Statement...")
        cash_flow_statement = process_cash_flow_statement(tables)

        apple_financials = AppleFinancials(
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            cash_flow_statement=cash_flow_statement
        )

        logging.info("Saving financial data to JSON...")
        save_financials(apple_financials, OUTPUT_PATH)

        format_results(apple_financials)
        logging.info(f"\nFull financial analysis saved to: {OUTPUT_PATH}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
