import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

batches = client.batches.list(limit=5)
for b in batches.data:
    print(f"Batch {b.id}: {b.status}")
    if b.errors:
        print(f"  Errors: {b.errors}")
