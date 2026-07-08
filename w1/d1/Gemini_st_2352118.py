import sys
from google import genai

client = genai.Client()

prompt = input("Enter your prompt: ")

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print("\nGemini answer:\n")
print(response.text)