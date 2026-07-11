import os
import sys
import time
import random

# =========================================================
# Import Gemini package
# =========================================================

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as gemini_errors

except ImportError:
    genai = None
    types = None
    gemini_errors = None


# =========================================================
# Import OpenAI package for LM Studio
# =========================================================

try:
    from openai import OpenAI
    from openai import APIConnectionError
    from openai import APIStatusError

except ImportError:
    OpenAI = None
    APIConnectionError = None
    APIStatusError = None


# =========================================================
# General settings
# =========================================================

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

gemini_client = None
gemini_chat = None

lm_client = None
lm_messages = []

model_name = None
provider_choice = None
system_instruction = None
temperature_value = 1.0
max_tokens_value = 1500


# =========================================================
# Choose system instruction
# =========================================================

def choose_system_instruction():
    print("\nChoose a system instruction:")
    print("1. General-purpose assistant")
    print("2. AI tutor")
    print("3. Coding assistant")
    print("4. Friendly chat companion")
    print("5. Study coach")
    print("6. Professional writing assistant")
    print("7. Concise assistant")
    print("8. Write a custom system instruction")

    choice = input("\nChoose option 1 to 8: ").strip()

    instructions = {
        "1": (
            "You are a helpful general-purpose AI assistant. "
            "Answer accurately, clearly, and practically. "
            "Answer in the same language as the user."
        ),

        "2": (
            "You are an expert AI Engineering tutor. "
            "Teach concepts clearly and step by step. "
            "Use simple examples before advanced explanations. "
            "Explain mistakes and how to correct them. "
            "Answer in the same language as the user."
        ),

        "3": (
            "You are an expert programming assistant. "
            "Help the user write, understand, debug, and improve code. "
            "Provide complete usable code when appropriate. "
            "Explain errors clearly and step by step. "
            "Use secure and maintainable programming practices. "
            "Answer in the same language as the user."
        ),

        "4": (
            "You are a friendly and supportive chat companion. "
            "Speak naturally, casually, and respectfully. "
            "Be warm and engaging without being overly formal. "
            "Answer in the same language as the user."
        ),

        "5": (
            "You are a study coach and teaching assistant. "
            "Help the user understand lessons, prepare for exams, "
            "organize study sessions, and practice difficult topics. "
            "Explain concepts step by step and use examples. "
            "Answer in the same language as the user."
        ),

        "6": (
            "You are a professional writing assistant. "
            "Help the user write, rewrite, summarize, translate, "
            "proofread, and improve professional and academic text. "
            "Preserve the intended meaning and use clear language. "
            "Answer in the same language as the user."
        ),

        "7": (
            "You are a concise AI assistant. "
            "Give direct and accurate answers without unnecessary details. "
            "Use short explanations unless the user asks for more detail. "
            "Answer in the same language as the user."
        )
    }

    if choice in instructions:
        selected_instruction = instructions[choice]

        print("\nSelected system instruction:")
        print(selected_instruction)

        return selected_instruction

    if choice == "8":
        print("\nWrite your custom system instruction.")
        print("You can use multiple lines.")
        print("Type END on a separate line when finished.\n")

        custom_lines = []

        while True:
            line = input()

            if line.strip().upper() == "END":
                break

            custom_lines.append(line)

        custom_instruction = "\n".join(custom_lines).strip()

        if not custom_instruction:
            print("\nNo instruction entered.")
            print("Using the general-purpose instruction.")

            return instructions["1"]

        print("\nCustom system instruction saved.")

        return custom_instruction

    print("\nInvalid choice.")
    print("Using the general-purpose instruction.")

    return instructions["1"]


# =========================================================
# Choose provider
# =========================================================

def choose_provider():
    print("Choose an AI provider:")
    print("1. Gemini API")
    print("2. LM Studio Local Server")

    while True:
        choice = input("\nChoose option 1 or 2: ").strip()

        if choice in ("1", "2"):
            return choice

        print("Invalid choice. Enter 1 or 2.")


# =========================================================
# Temperature setup
# =========================================================

def choose_temperature():
    if provider_choice == "1":
        default_temperature = 1.0
    else:
        default_temperature = 0.7

    print("\nTemperature controls randomness and creativity.")
    print(f"Recommended default: {default_temperature}")

    keep_default = input(
        f"Keep temperature at {default_temperature}? (yes/no): "
    ).strip().lower()

    if keep_default in ("yes", "y", ""):
        return default_temperature

    try:
        value = float(
            input("Enter temperature value: ").strip()
        )

        if value < 0:
            raise ValueError

        return value

    except ValueError:
        print(
            f"Invalid value. Using default: {default_temperature}"
        )

        return default_temperature


# =========================================================
# Maximum token setup
# =========================================================

def choose_max_tokens():
    print("\nChoose maximum output tokens:")
    print("1. Short answers: 700")
    print("2. Normal explanations: 1500")
    print("3. Coding and debugging: 2500")
    print("4. Deep explanations: 4000")
    print("5. Long writing: 3000")
    print("6. Custom value")

    choice = input("\nChoose option 1 to 6: ").strip()

    token_options = {
        "1": 700,
        "2": 1500,
        "3": 2500,
        "4": 4000,
        "5": 3000
    }

    if choice in token_options:
        return token_options[choice]

    if choice == "6":
        try:
            value = int(
                input("Enter maximum output tokens: ").strip()
            )

            if value <= 0:
                raise ValueError

            return value

        except ValueError:
            print("Invalid value. Using 1500.")
            return 1500

    print("Invalid choice. Using 1500.")
    return 1500


# =========================================================
# Gemini functions
# =========================================================

def create_gemini_chat():
    return gemini_client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            temperature=temperature_value,
            max_output_tokens=max_tokens_value,
            system_instruction=system_instruction
        )
    )


def setup_gemini():
    global gemini_client
    global gemini_chat
    global model_name

    if genai is None:
        print("\nThe google-genai package is not installed.")
        print("Install it using:")
        print("py -m pip install --upgrade google-genai")
        sys.exit(1)

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMINI_KEY")
    )

    if not api_key:
        print("\nGemini API key was not found.")
        print("Save it as GEMINI_API_KEY or GEMINI_KEY.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

    print("\nChoose a Gemini model:")
    print("1. gemini-3.5-flash")
    print("2. Enter a custom model name")

    choice = input("\nChoose option 1 or 2: ").strip()

    if choice == "1":
        model_name = "gemini-3.5-flash"

    elif choice == "2":
        custom_name = input(
            "Enter the model name: "
        ).strip()

        model_name = custom_name or "gemini-3.5-flash"

    else:
        print("Invalid choice. Using gemini-3.5-flash.")
        model_name = "gemini-3.5-flash"

    try:
        gemini_chat = create_gemini_chat()

    except Exception as error:
        print("\nCould not create the Gemini chat.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")
        sys.exit(1)


def send_gemini_message(prompt, max_attempts=5):
    retryable_codes = {
        408,
        429,
        500,
        502,
        503,
        504
    }

    for attempt in range(1, max_attempts + 1):
        try:
            return gemini_chat.send_message(prompt)

        except gemini_errors.APIError as error:
            error_code = getattr(error, "code", None)
            error_message = getattr(
                error,
                "message",
                str(error)
            )

            if (
                error_code in retryable_codes
                and attempt < max_attempts
            ):
                wait_seconds = (
                    2 ** (attempt - 1)
                    + random.uniform(0, 1)
                )

                print("\nTemporary Gemini API error.")
                print(f"Error code: {error_code}")
                print(f"Message: {error_message}")

                print(
                    f"Retrying in {wait_seconds:.1f} seconds "
                    f"({attempt + 1}/{max_attempts})..."
                )

                time.sleep(wait_seconds)
                continue

            print("\nGemini API request failed.")
            print(f"Error code: {error_code}")
            print(f"Message: {error_message}")

            return None

        except KeyboardInterrupt:
            print("\nRequest cancelled.")
            return None

        except Exception as error:
            print("\nUnexpected Gemini error.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error: {error}")

            return None

    return None


# =========================================================
# LM Studio functions
# =========================================================

def reset_lm_studio_messages():
    global lm_messages

    lm_messages = [
        {
            "role": "system",
            "content": system_instruction
        }
    ]


def setup_lm_studio():
    global lm_client
    global model_name

    if OpenAI is None:
        print("\nThe openai package is not installed.")
        print("Install it using:")
        print("py -m pip install --upgrade openai")
        sys.exit(1)

    lm_client = OpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio"
    )

    try:
        models_response = lm_client.models.list()

    except APIConnectionError:
        print("\nCould not connect to LM Studio.")
        print("Open LM Studio and start the Local Server.")
        print(f"Expected address: {LM_STUDIO_BASE_URL}")
        sys.exit(1)

    except Exception as error:
        print("\nCould not read LM Studio models.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")
        sys.exit(1)

    model_ids = [
        model.id
        for model in models_response.data
    ]

    if not model_ids:
        print("\nNo models were found in LM Studio.")
        print("Load a model and start the server.")
        sys.exit(1)

    print("\nAvailable LM Studio models:")

    for index, local_model in enumerate(
        model_ids,
        start=1
    ):
        print(f"{index}. {local_model}")

    print(f"{len(model_ids) + 1}. Enter a custom model ID")

    choice = input("\nChoose a model: ").strip()

    try:
        selected_number = int(choice)

        if 1 <= selected_number <= len(model_ids):
            model_name = model_ids[selected_number - 1]

        elif selected_number == len(model_ids) + 1:
            model_name = input(
                "Enter the model ID: "
            ).strip()

            if not model_name:
                raise ValueError

        else:
            raise ValueError

    except ValueError:
        model_name = model_ids[0]

        print(
            f"Invalid choice. Using the first model: "
            f"{model_name}"
        )

    reset_lm_studio_messages()


def send_lm_studio_message(prompt, max_attempts=3):
    global lm_messages

    user_message = {
        "role": "user",
        "content": prompt
    }

    lm_messages.append(user_message)

    retryable_codes = {
        408,
        429,
        500,
        502,
        503,
        504
    }

    for attempt in range(1, max_attempts + 1):
        try:
            response = lm_client.chat.completions.create(
                model=model_name,
                messages=lm_messages,
                temperature=temperature_value,
                max_tokens=max_tokens_value
            )

            answer = response.choices[0].message.content

            if not answer:
                print("\nLM Studio returned an empty response.")

                finish_reason = (
                    response.choices[0].finish_reason
                )

                if finish_reason:
                    print(f"Finish reason: {finish_reason}")

                lm_messages.pop()
                return None

            lm_messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            return response

        except APIConnectionError:
            print("\nConnection to LM Studio was lost.")
            print("Make sure the Local Server is running.")

            lm_messages.pop()
            return None

        except APIStatusError as error:
            status_code = error.status_code

            if (
                status_code in retryable_codes
                and attempt < max_attempts
            ):
                wait_seconds = 2 ** (attempt - 1)

                print("\nTemporary LM Studio error.")
                print(f"Status code: {status_code}")

                print(
                    f"Retrying in {wait_seconds} seconds "
                    f"({attempt + 1}/{max_attempts})..."
                )

                time.sleep(wait_seconds)
                continue

            print("\nLM Studio API request failed.")
            print(f"Status code: {status_code}")
            print(f"Message: {error.message}")

            lm_messages.pop()
            return None

        except KeyboardInterrupt:
            print("\nRequest cancelled.")

            lm_messages.pop()
            return None

        except Exception as error:
            print("\nUnexpected LM Studio error.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error: {error}")

            lm_messages.pop()
            return None

    lm_messages.pop()
    return None


# =========================================================
# Change system instruction during chat
# =========================================================

def change_system_instruction():
    global system_instruction
    global gemini_chat

    system_instruction = choose_system_instruction()

    if provider_choice == "1":
        gemini_chat = create_gemini_chat()
    else:
        reset_lm_studio_messages()

    print("\nSystem instruction changed.")
    print("Conversation memory was cleared.\n")


# =========================================================
# Initial setup
# =========================================================

provider_choice = choose_provider()
system_instruction = choose_system_instruction()
temperature_value = choose_temperature()
max_tokens_value = choose_max_tokens()

if provider_choice == "1":
    provider_name = "Gemini API"
    setup_gemini()

else:
    provider_name = "LM Studio Local Server"
    setup_lm_studio()


# =========================================================
# Display chat information
# =========================================================

print("\n====================================")
print("Chat started")
print("====================================")
print(f"Provider: {provider_name}")
print(f"Model: {model_name}")
print(f"Temperature: {temperature_value}")
print(f"Maximum output tokens: {max_tokens_value}")

print("\nCommands:")
print("clear       Clear conversation memory")
print("system      Change the system instruction")
print("instruction Show the current system instruction")
print("exit        Close the program")
print()


# =========================================================
# Main chat loop
# =========================================================

try:
    while True:
        prompt = input("You: ").strip()

        if not prompt:
            continue

        command = prompt.lower()

        if command == "exit":
            print("Chat ended.")
            break

        if command == "instruction":
            print("\nCurrent system instruction:")
            print(system_instruction)
            print()
            continue

        if command == "system":
            change_system_instruction()
            continue

        if command == "clear":
            if provider_choice == "1":
                gemini_chat = create_gemini_chat()
            else:
                reset_lm_studio_messages()

            print("Conversation memory cleared.\n")
            continue

        start_time = time.perf_counter()

        # -------------------------------------------------
        # Gemini
        # -------------------------------------------------

        if provider_choice == "1":
            response = send_gemini_message(prompt)

            time_passed = (
                time.perf_counter() - start_time
            )

            if response is None:
                print("\nYou can enter another prompt.\n")
                continue

            print("\nGemini answer:\n")

            if response.text:
                print(response.text)

            else:
                print("[Gemini returned no visible text.]")

                if response.candidates:
                    candidate = response.candidates[0]

                    print(
                        f"Finish reason: "
                        f"{candidate.finish_reason}"
                    )

            usage = response.usage_metadata

            print("\n--- Response information ---")
            print(f"Time passed: {time_passed:.2f} Sec")

            if usage:
                print(
                    f"Input tokens: "
                    f"{usage.prompt_token_count}"
                )

                print(
                    f"Output tokens: "
                    f"{usage.candidates_token_count}"
                )

                thinking_tokens = getattr(
                    usage,
                    "thoughts_token_count",
                    None
                )

                if thinking_tokens is not None:
                    print(
                        f"Thinking tokens: "
                        f"{thinking_tokens}"
                    )

                print(
                    f"Total tokens: "
                    f"{usage.total_token_count}"
                )

            else:
                print("Token information is unavailable.")

        # -------------------------------------------------
        # LM Studio
        # -------------------------------------------------

        else:
            response = send_lm_studio_message(prompt)

            time_passed = (
                time.perf_counter() - start_time
            )

            if response is None:
                print("\nYou can enter another prompt.\n")
                continue

            answer = response.choices[0].message.content

            print("\nLocal LLM answer:\n")
            print(answer)

            print("\n--- Response information ---")
            print(f"Time passed: {time_passed:.2f} Sec")

            if response.usage:
                print(
                    f"Input tokens: "
                    f"{response.usage.prompt_tokens}"
                )

                print(
                    f"Output tokens: "
                    f"{response.usage.completion_tokens}"
                )

                print(
                    f"Total tokens: "
                    f"{response.usage.total_tokens}"
                )

            else:
                print("Token information is unavailable.")

        print()


except KeyboardInterrupt:
    print("\nChat ended by the user.")


finally:
    try:
        if gemini_client is not None:
            gemini_client.close()

        if lm_client is not None:
            lm_client.close()

    except Exception:
        pass