import os
from dotenv import load_dotenv
from openai import OpenAI

# Load your .env file
load_dotenv()

# Read credentials from .env
host_target = os.getenv("DATABRICKS_HOST")
auth_token = os.getenv("DATABRICKS_TOKEN")
# model_endpoint = os.getenv("MODEL_ENDPOINT_NAME")
model_endpoint = os.getenv("DATABRICKS_MODEL")

# Make sure all values are set
if not all([host_target, auth_token, model_endpoint]):
    raise ValueError("Check your .env file — one or more values are missing")

print("Connecting to Databricks Mosaic AI...")
print(f"Model: {model_endpoint}")

# Connect using the OpenAI-compatible API
client = OpenAI(
    base_url = f"{host_target}/serving-endpoints",
    api_key = auth_token
)

# Send a test message
try:
    response = client.chat.completions.create(
        model = model_endpoint,
        messages = [{"role": "user","content": "Confirm that my local Python can talk to you!"}],
        temperature = 0.3
    )
    print("\n===== SUCCESS =====")
    print(response.choices[0].message.content)
    print("===================")

except Exception as e:
    print(f"Connection failed: {e}")
    print("Check your DATABRICKS_HOST and DATABRICKS_TOKEN in .env")