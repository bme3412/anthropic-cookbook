# File: scripts/process_transcripts.py

import json
import os
from textwrap import dedent
from typing import List
from models import TranscriptSchema
from utils import list_transcript_files, load_transcript
from pydantic import ValidationError
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(
    filename='process_transcripts.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

MODEL='gpt-4o-mini'
client = OpenAI()

# Initialize the OpenAI API client
client.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set as an environment variable

# Define the summarization prompt tailored for earnings call transcripts
summarization_prompt = '''
You will be provided with a transcript of an earnings call.
Your task is to extract and summarize the following information in a structured JSON format:
- ticker: The stock ticker symbol of the company.
- download_date: The date when the transcript was downloaded.
- call_date: The date of the earnings call.
- start_time: The start time of the call.
- participants:
    - operator: Name and role.
    - investor_relations: Name and title.
    - executives: List of executives with names and titles.
- disclaimers:
    - recording_notice: Notice about the call being recorded.
    - forward_looking_statements: Text and reference page.
- presentation_materials:
    - availability_date: When materials were posted.
    - url: Link to access materials.
    - description: Description of materials' availability.
- call_details:
    - duration_minutes: Total duration of the call.
    - qa_instructions: Instructions for the Q&A session.
- executive_comments: List of comments with speaker name, timestamp, and content.
- financial_highlights:
    - quarter: Fiscal quarter reported.
    - sales: Amount, currency, growth metrics, and notes.
    - profitability: Summary of profitability and cash flow.
    - regional_performance: Details on regional sales performance.
- qna_session:
    - scheduled: Whether the Q&A session is scheduled.
    - details:
        - after_executives: Boolean indicating if Q&A is after executives.
        - duration_minutes: Duration allocated for Q&A.
        - rules: Rules for participants during Q&A.
Ensure the JSON is well-structured and follows the schema provided.
'''

def get_structured_transcript(transcript_text: str) -> dict:
    """
    Integrates with the OpenAI API to parse the transcript into a structured JSON format.
    """
    try:
        response = client.beta.chat.completions.parse(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that structures earnings call transcripts."},
                {"role": "user", "content": dedent(summarization_prompt) + "\n\nTranscript:\n" + transcript_text}
            ],
            max_tokens=3000,  # Adjust based on the expected length of the response
        )
        
        # Extract the content from the response
        structured_json_str = response.choices[0].message.parsed
        
        # Parse the JSON string into a dictionary
        structured_json = json.loads(structured_json_str)
        
        return structured_json
    
    except Exception as e:
        print(f"Error during API call: {e}")
        return {}

def main():
    transcript_dir = "../data/earnings_calls"
    output_dir = "../data/structured_outputs_earnings_calls/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    transcript_files = list_transcript_files(transcript_dir)
    print(f"Found {len(transcript_files)} transcript files.")
    
    structured_transcripts: List[dict] = []
    
    for idx, filepath in enumerate(transcript_files):
        filename = os.path.basename(filepath)
        print(f"\nProcessing file {idx+1}/{len(transcript_files)}: {filename}")
        transcript_text = load_transcript(filepath)
        
        # Parse the transcript using the language model
        structured_data = get_structured_transcript(transcript_text)
        
        if not structured_data:
            print(f"Skipping {filename} due to previous errors.")
            continue
        
        # Validate and parse using Pydantic
        try:
            transcript_schema = TranscriptSchema(**structured_data)
            structured_transcripts.append(transcript_schema.dict())
            print(f"Successfully processed {filename}.")
            
            # Save the structured data to a JSON file
            output_filename = f"{transcript_schema.ticker}_{transcript_schema.transcript.call_date.replace('-', '')}.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(transcript_schema.dict(), outfile, indent=2)
            print(f"Saved structured data to {output_filename}.")
            
        except ValidationError as ve:
            print(f"Validation Error for {filename}: {ve}")
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
    
    # Optionally, save all structured transcripts into a single JSON file
    all_output_path = os.path.join(output_dir, "structured_earnings_calls.json")
    with open(all_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(structured_transcripts, outfile, indent=2)
    print(f"\nAll structured transcripts saved to {all_output_path}.")

if __name__ == "__main__":
    main()
