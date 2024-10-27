-- create_tables.sql

-- companies table
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(10) UNIQUE NOT NULL
);

-- transcripts table
CREATE TABLE IF NOT EXISTS transcripts (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id),
    earnings_date TIMESTAMP NOT NULL,
    quarter INTEGER NOT NULL,
    s3_key VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- transcript_sections table
CREATE TABLE IF NOT EXISTS transcript_sections (
    id SERIAL PRIMARY KEY,
    transcript_id INTEGER REFERENCES transcripts(id),
    speaker_name VARCHAR(255),
    speaker_role VARCHAR(255),
    text TEXT,
    timestamp TIMESTAMP
);
