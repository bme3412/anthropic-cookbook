# app.py
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
import os

from analyzer import TranscriptAnalyzer, AnalyzerException
from utils import save_analysis_to_json, load_analysis_from_json

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')

# Add template context processor to make datetime available to all templates
@app.context_processor
def utility_processor():
    return {
        'datetime': datetime,
        'current_year': datetime.now().year
    }

def create_analyzer():
    """Create analyzer instance with error handling"""
    try:
        return TranscriptAnalyzer(max_retries=3, retry_delay=5)
    except AnalyzerException as e:
        print(f"Failed to create analyzer: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').upper()
        quarter = request.form.get('quarter')
        year = request.form.get('year')

        # Input validation
        if not symbol:
            flash('Please enter a valid stock symbol.', 'danger')
            return redirect(url_for('index'))
        if not quarter or not quarter.isdigit() or not (1 <= int(quarter) <= 4):
            flash('Quarter must be an integer between 1 and 4.', 'danger')
            return redirect(url_for('index'))
        if not year or not year.isdigit() or not (2000 <= int(year) <= datetime.now().year):
            flash(f'Year must be an integer between 2000 and {datetime.now().year}.', 'danger')
            return redirect(url_for('index'))

        quarter = int(quarter)
        year = int(year)

        try:
            analyzer = create_analyzer()
            if not analyzer:
                flash('Failed to initialize analyzer. Please check your configuration.', 'danger')
                return redirect(url_for('index'))

            analysis, usage_stats = analyzer.analyze_transcript(symbol=symbol, quarter=quarter, year=year)
            
            filename = save_analysis_to_json(
                analysis, 
                usage_stats, 
                filename=f"{symbol}_analysis_Q{quarter}_{year}.json"
            )
            
            if filename:
                flash('Analysis completed successfully.', 'success')
                return redirect(url_for('results', filename=filename))
            else:
                flash('Failed to save analysis results.', 'danger')
                return redirect(url_for('index'))

        except AnalyzerException as e:
            flash(f"Analysis failed: {str(e)}", 'danger')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Unexpected error: {str(e)}", 'danger')
            return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    try:
        data = load_analysis_from_json(filename)
        if data:
            return render_template('results.html', data=data)
        flash('Failed to load analysis results.', 'danger')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error loading results: {str(e)}", 'danger')
        return redirect(url_for('index'))

def cli_main():
    try:
        analyzer = create_analyzer()
        if not analyzer:
            print("Failed to initialize analyzer. Please check your configuration.")
            return

        analysis, usage_stats = analyzer.analyze_transcript(
            symbol=args.symbol,
            quarter=args.quarter,
            year=args.year
        )

        filename = save_analysis_to_json(
            analysis,
            usage_stats,
            filename=f"{args.symbol}_analysis_Q{args.quarter}_{args.year}.json"
        )

        if filename:
            print(f"\nAnalysis saved to {filename}")
            data = load_analysis_from_json(filename)
            if data:
                print("\nAnalysis Results:")
                print(f"Company: {data['company']}")
                print(f"Period: Q{data['quarter']} {data['year']}")
                print("\nSummary:")
                print(data['summary'])
                print("\nKey Highlights:")
                for highlight in data['key_highlights']:
                    print(f"- {highlight}")
                print("\nUsage Statistics:")
                print(f"Total Tokens: {data['usage_statistics']['total_tokens']:,}")
                print(f"Total Cost: ${data['usage_statistics']['total_cost']:.4f}")

    except AnalyzerException as e:
        print(f"Analysis failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings Call Transcript Analyzer")
    parser.add_argument('--flask', action='store_true', help='Run Flask web app')
    parser.add_argument('--symbol', type=str, help='Stock symbol')
    parser.add_argument('--quarter', type=int, help='Quarter (1-4)')
    parser.add_argument('--year', type=int, help='Year')

    args = parser.parse_args()

    if args.flask or not any([args.symbol, args.quarter, args.year]):
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        cli_main()