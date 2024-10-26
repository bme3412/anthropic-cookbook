from dotenv import load_dotenv
from anthropic import Anthropic

#load environment variable
load_dotenv()

#automatically looks for an "ANTHROPIC_API_KEY" environment variable
client = Anthropic()

truncated_response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=50,
    messages=[
        {"role": "user", "content": "Write me a poem about Cape Cod"}
    ]
)
print(truncated_response.content[0].text)

response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Tell me a joke about Cape Cod"}]
)

print(response.content[0].text)
print(response.usage.output_tokens)

response_json = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=2500,
    messages=[{"role": "user", "content": "Generate a JSON object representing the town of Barnstable, list of attractions, and best beaches in Barnstable"}],
    
)
print(response_json.content[0].text)
print(response_json.usage.output_tokens)