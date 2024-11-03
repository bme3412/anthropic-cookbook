# app.py
from flask import Flask, render_template
from routes import api_bp
import os
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import sys
import nltk
from pathlib import Path

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLTK once at startup
def initialize_nltk():
    """Initialize NLTK data before starting the app"""
    try:
        nltk_data_dir = str(Path.home() / 'nltk_data' / 'tokenizers' / 'punkt')
        if not os.path.exists(nltk_data_dir):
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        else:
            logger.info("NLTK punkt tokenizer already downloaded")
    except Exception as e:
        logger.error(f"Error initializing NLTK: {str(e)}")
        raise

# Initialize NLTK at module level
initialize_nltk()

def create_app():
    logger.info("Creating Flask application...")
    
    app = Flask(__name__)
    CORS(app)
    
    # Configure Flask app
    app.config.update(
        SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'dev'),
        JSON_SORT_KEYS=False,
        TEMPLATES_AUTO_RELOAD=True,
        SEND_FILE_MAX_AGE_DEFAULT=0
    )
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    logger.info("Registered API blueprint")
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.errorhandler(404)
    def not_found_error(error):
        logger.error(f"404 error: {error}")
        return render_template('index.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 error: {error}")
        return render_template('index.html'), 500
    
    logger.info("Flask application created successfully")
    return app

if __name__ == '__main__':
    try:
        app = create_app()
        port = int(os.getenv('PORT', 5000))
        
        print(f"\n{'='*50}")
        print(f"Server is running on http://localhost:{port}")
        print(f"Press Ctrl+C to stop the server")
        print(f"{'='*50}\n")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
