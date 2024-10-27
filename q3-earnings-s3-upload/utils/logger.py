# utils/logger.py

import logging

# Create a custom logger
logger = logging.getLogger("earnings_transcripts")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('app.log')

c_handler.setLevel(logging.DEBUG)  # Set to DEBUG
f_handler.setLevel(logging.DEBUG)  # Set to DEBUG

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
