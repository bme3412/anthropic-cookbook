# storage/db_storage.py

import psycopg2
from psycopg2.extras import RealDictCursor
from utils.config import DATABASE_URL
from utils.logger import logger

def get_db_connection():
    """
    Establishes and returns a new database connection.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def insert_company(name: str, ticker: str):
    """
    Inserts a company into the database if it doesn't exist.
    Returns the company ID.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO companies (name, ticker)
                VALUES (%s, %s)
                ON CONFLICT (ticker) DO NOTHING
                RETURNING id;
            """, (name, ticker))
            result = cur.fetchone()
            if result:
                company_id = result['id']
            else:
                # Fetch existing company id
                cur.execute("SELECT id FROM companies WHERE ticker = %s;", (ticker,))
                existing = cur.fetchone()
                company_id = existing['id'] if existing else None
            conn.commit()
            return company_id
    except Exception as e:
        logger.error(f"Error inserting company: {e}")
    finally:
        conn.close()

def insert_transcript(company_id: int, earnings_date: str, quarter: int, s3_key: str):
    """
    Inserts a transcript into the database.
    Returns the transcript ID.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            # Check if transcript already exists
            cur.execute("""
                SELECT id FROM transcripts
                WHERE company_id = %s AND earnings_date = %s AND quarter = %s;
            """, (company_id, earnings_date, quarter))
            existing = cur.fetchone()
            if existing:
                logger.info(f"Transcript already exists with ID {existing['id']}")
                return existing['id']
            
            # Insert new transcript
            cur.execute("""
                INSERT INTO transcripts (company_id, earnings_date, quarter, s3_key)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (company_id, earnings_date, quarter, s3_key))
            transcript_id = cur.fetchone()['id']
            conn.commit()
            return transcript_id
    except Exception as e:
        logger.error(f"Error inserting transcript: {e}")
    finally:
        conn.close()

def insert_transcript_section(transcript_id: int, speaker_name: str, speaker_role: str, text: str, timestamp: str):
    """
    Inserts a transcript section into the database.
    """
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transcript_sections (transcript_id, speaker_name, speaker_role, text, timestamp)
                VALUES (%s, %s, %s, %s, %s);
            """, (transcript_id, speaker_name, speaker_role, text, timestamp))
            conn.commit()
    except Exception as e:
        logger.error(f"Error inserting transcript section: {e}")
    finally:
        conn.close()
