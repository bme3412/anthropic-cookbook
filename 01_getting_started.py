from dotenv import load_dotenv
import os
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Initialize the client with your API key
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")  # Make sure this environment variable is set
)

our_first_message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hi there! Please write me a haiku about Cape Cod"}
    ]
)

print(our_first_message.content[0].text)