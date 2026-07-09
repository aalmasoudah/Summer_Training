import os
import sys
import time
from google import genai
from google.genai import types


# -----------------------------
# API key setup
# -----------------------------
# This tries GEMINI_API_KEY first, then GEMINI_KEY.
# Use the one you saved in Windows environment variables.
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")

if not api_key:
    print("API key is missing.")
    print("Make sure you saved it as GEMINI_API_KEY or GEMINI_KEY.")
    sys.exit(1)

client = genai.Client(api_key=api_key)


# -----------------------------
# Model setup
# -----------------------------
print("Choose a Gemini model:")
print("1. gemini-3.5-flash")
print("2. Type a custom model name")

model_choice = input("Choose option 1 or 2: ")

if model_choice == "1":
    model_name = "gemini-3.5-flash"
elif model_choice == "2":
    model_name = input("Enter model name: ")
else:
    print("Invalid choice. Using default model: gemini-3.5-flash")
    model_name = "gemini-3.5-flash"


# -----------------------------
# Temperature setup
# -----------------------------
print("\nTemperature controls how random or creative the model's answers are.")
print("You can change the temperature, but Google recommends keeping it at 1.0 for Gemini 3.x models like gemini-3.5-flash.")
print("For most uses, especially studying, coding, and reasoning, keep it at 1.0.\n")

choice = input("Do you want to keep temperature at 1.0? (yes/no): ").lower()

if choice == "yes" or choice == "y":
    temperature_value = 1.0
else:
    temperature_value = float(input("Enter temperature value: "))


# -----------------------------
# Output token setup
# -----------------------------
print("\nMaximum output tokens controls how long the model's answer can be.")
print("A higher number allows longer answers, but it may be slower and use more tokens.")
print("A lower number makes answers shorter, but the answer may get cut off if it is too small.\n")

print("Recommended max_output_tokens values:")
print("1. Short answers / quick questions: 200")
print("2. Normal studying explanation: 700")
print("3. Coding help / debugging: 1200")
print("4. Deep reasoning / step-by-step explanation: 2000")
print("5. Creative writing / brainstorming: 1500")
print("6. Custom value")

token_choice = input("\nChoose an option from 1 to 6: ")

if token_choice == "1":
    max_tokens_value = 200
elif token_choice == "2":
    max_tokens_value = 700
elif token_choice == "3":
    max_tokens_value = 1200
elif token_choice == "4":
    max_tokens_value = 2000
elif token_choice == "5":
    max_tokens_value = 1500
elif token_choice == "6":
    max_tokens_value = int(input("Enter maximum output tokens: "))
else:
    print("Invalid choice. Using default value: 700")
    max_tokens_value = 700


# -----------------------------
# Create chat
# -----------------------------
chat = client.chats.create(
    model=model_name,
    config=types.GenerateContentConfig(
        temperature=temperature_value,
        max_output_tokens=max_tokens_value,
        system_instruction="You are a helpful AI Engineering tutor. Explain clearly and step by step."
    )
)

print("\nChat started.")
print(f"Model: {model_name}")
print(f"Temperature: {temperature_value}")
print(f"Maximum output tokens: {max_tokens_value}")
print("Type 'exit' to stop.\n")


# -----------------------------
# Chat loop
# -----------------------------
while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        print("Chat ended.")
        break

    try:
        start_time = time.perf_counter()

        response = chat.send_message(prompt)

        end_time = time.perf_counter()
        time_passed = end_time - start_time

        print("\nGemini answer:\n")
        print(response.text)

        usage = response.usage_metadata

        print("\n--- Response info ---")
        print(f"Time passed: {time_passed:.2f} Sec")

        if usage:
            print(f"Tokens used: {usage.total_token_count}")
            print(f"Input tokens: {usage.prompt_token_count}")
            print(f"Output tokens: {usage.candidates_token_count}")
        else:
            print("Tokens used: Not available")

        print()

    except Exception as error:
        print("\nSomething went wrong.")
        print("Error:")
        print(error)
        print()