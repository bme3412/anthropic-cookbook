### BASIC SETUP

from dotenv import load_dotenv
from anthropic import Anthropic

#load environment variable
load_dotenv()

#automatically looks for an "ANTHROPIC_API_KEY" environment variable
client = Anthropic()

response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Translate 'hello, how are you?' to French, Spanish, Italian, Portuguese. Respond with an appropriate phase for each"}
    ]
)

print(response.content[0].text)