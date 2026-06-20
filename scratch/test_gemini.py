import os
import asyncio
from google.genai import Client
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Set GEMINI_API_KEY if GOOGLE_API_KEY is present
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

print("GEMINI_API_KEY set:", "GEMINI_API_KEY" in os.environ)

async def test_call():
    client = Client()
    # Let's try calling the requested model: gemini-3.1-flash-lite
    model_name = "gemini-3.1-flash-lite"
    print(f"Calling model: {model_name}")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say hello and confirm you are online.",
        )
        print("Response text:", response.text)
    except Exception as e:
        print("Failed with model gemini-3.1-flash-lite:", e)
        # Let's try gemini-2.5-flash as fallback in case gemini-3.1-flash-lite is not enabled or available in our credentials
        print("Trying fallback gemini-2.5-flash...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Say hello and confirm you are online.",
            )
            print("Response text:", response.text)
        except Exception as ex:
            print("Failed with fallback too:", ex)

asyncio.run(test_call())
