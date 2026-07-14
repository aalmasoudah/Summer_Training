import os
import sys
import time
import random
import html

from pathlib import Path
from datetime import datetime


# =========================================================
# Optional imports: Gemini
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
# Optional imports: OpenAI client for LM Studio
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
# General configuration
# =========================================================

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

gemini_client = None
gemini_chat = None

lm_client = None
lm_messages = []

provider_choice = None
provider_name = None
model_name = None

system_instruction = None
temperature_value = 1.0
max_tokens_value = 1500

conversation_log = []

session_started = datetime.now()
session_id = session_started.strftime("%Y-%m-%d_%H-%M-%S")

try:
    script_directory = Path(__file__).resolve().parent
except NameError:
    script_directory = Path.cwd()

export_directory = script_directory / "chat_exports"
export_directory.mkdir(parents=True, exist_ok=True)

txt_export_path = export_directory / f"chat_{session_id}.txt"
pdf_export_path = export_directory / f"chat_{session_id}.pdf"

export_mode = "manual"


# =========================================================
# Conversation logging
# =========================================================

def record_message(role, content):
    """
    Add a message to the permanent conversation log.

    Roles:
    - system
    - user
    - assistant
    """

    conversation_log.append(
        {
            "role": role,
            "content": str(content),
            "time": datetime.now()
        }
    )


def role_display_name(role):
    if role == "user":
        return "You"

    if role == "assistant":
        if provider_choice == "1":
            return "Gemini"

        return "Local LLM"

    return "System"


# =========================================================
# TXT export
# =========================================================

def export_chat_to_txt():
    """
    Write the full conversation to a UTF-8 text file.
    """

    try:
        with open(
            txt_export_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write("AI CHAT TRANSCRIPT\n")
            file.write("=" * 70 + "\n\n")

            file.write(
                f"Session started: "
                f"{session_started.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

            file.write(f"Provider: {provider_name}\n")
            file.write(f"Model: {model_name}\n")
            file.write(f"Temperature: {temperature_value}\n")
            file.write(
                f"Maximum output tokens: {max_tokens_value}\n"
            )

            file.write("\nSystem instruction:\n")
            file.write(system_instruction)
            file.write("\n\n")
            file.write("=" * 70 + "\n\n")

            if not conversation_log:
                file.write("No conversation messages yet.\n")

            for message in conversation_log:
                timestamp = message["time"].strftime("%H:%M:%S")
                display_role = role_display_name(message["role"])

                file.write(
                    f"[{timestamp}] {display_role}\n"
                )

                file.write("-" * 70 + "\n")
                file.write(message["content"])
                file.write("\n\n")

        return True

    except OSError as error:
        print("\nCould not save the TXT file.")
        print(f"Error: {error}")
        return False


# =========================================================
# PDF support functions
# =========================================================

def contains_arabic(text):
    """
    Detect Arabic characters in a string.
    """

    arabic_ranges = (
        ("\u0600", "\u06FF"),
        ("\u0750", "\u077F"),
        ("\u08A0", "\u08FF"),
        ("\uFB50", "\uFDFF"),
        ("\uFE70", "\uFEFF")
    )

    for character in text:
        for beginning, ending in arabic_ranges:
            if beginning <= character <= ending:
                return True

    return False


def find_unicode_font():
    """
    Find a Windows font that supports English and Arabic.
    """

    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/aptos.ttf")
    ]

    for font_path in font_candidates:
        if font_path.exists():
            return font_path

    return None


def prepare_pdf_text(text):
    """
    Escape text for ReportLab and apply Arabic shaping.

    Returns:
        formatted_text
        has_arabic
    """

    text = str(text)
    has_arabic = contains_arabic(text)

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        arabic_support_available = True

    except ImportError:
        arabic_support_available = False

    prepared_lines = []

    for line in text.splitlines():
        if contains_arabic(line) and arabic_support_available:
            line = arabic_reshaper.reshape(line)
            line = get_display(line)

        prepared_lines.append(html.escape(line))

    if not prepared_lines:
        prepared_lines.append("")

    formatted_text = "<br/>".join(prepared_lines)

    return formatted_text, has_arabic


# =========================================================
# PDF export
# =========================================================

def export_chat_to_pdf():
    """
    Generate a PDF containing the full conversation.
    """

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            PageBreak
        )

    except ImportError:
        print("\nPDF libraries are missing.")
        print("Install them using:")
        print(
            "py -m pip install "
            "reportlab arabic-reshaper python-bidi"
        )
        return False

    font_path = find_unicode_font()

    if font_path is None:
        print("\nCould not find a suitable Windows font.")
        print(
            "The program searched for Arial, Segoe UI, "
            "Tahoma, Calibri, and Aptos."
        )
        return False

    try:
        pdfmetrics.registerFont(
            TTFont(
                "ChatUnicode",
                str(font_path)
            )
        )

        document = SimpleDocTemplate(
            str(pdf_export_path),
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title="AI Chat Transcript",
            author="AI Chat Program"
        )

        title_style = ParagraphStyle(
            name="Title",
            fontName="ChatUnicode",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=10
        )

        metadata_style = ParagraphStyle(
            name="Metadata",
            fontName="ChatUnicode",
            fontSize=9,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=3
        )

        role_style_left = ParagraphStyle(
            name="RoleLeft",
            fontName="ChatUnicode",
            fontSize=11,
            leading=14,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=4
        )

        role_style_right = ParagraphStyle(
            name="RoleRight",
            fontName="ChatUnicode",
            fontSize=11,
            leading=14,
            alignment=TA_RIGHT,
            spaceBefore=8,
            spaceAfter=4
        )

        message_style_left = ParagraphStyle(
            name="MessageLeft",
            fontName="ChatUnicode",
            fontSize=10,
            leading=15,
            alignment=TA_LEFT,
            spaceAfter=9
        )

        message_style_right = ParagraphStyle(
            name="MessageRight",
            fontName="ChatUnicode",
            fontSize=10,
            leading=15,
            alignment=TA_RIGHT,
            spaceAfter=9
        )

        story = []

        story.append(
            Paragraph(
                "AI Chat Transcript",
                title_style
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    "Session started: "
                    + session_started.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ),
                metadata_style
            )
        )

        story.append(
            Paragraph(
                html.escape(f"Provider: {provider_name}"),
                metadata_style
            )
        )

        story.append(
            Paragraph(
                html.escape(f"Model: {model_name}"),
                metadata_style
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    f"Temperature: {temperature_value}"
                ),
                metadata_style
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    "Maximum output tokens: "
                    f"{max_tokens_value}"
                ),
                metadata_style
            )
        )

        story.append(Spacer(1, 10))

        system_text, system_has_arabic = prepare_pdf_text(
            system_instruction
        )

        system_heading = "System Instruction"

        story.append(
            Paragraph(
                system_heading,
                role_style_left
            )
        )

        story.append(
            Paragraph(
                system_text,
                (
                    message_style_right
                    if system_has_arabic
                    else message_style_left
                )
            )
        )

        story.append(Spacer(1, 8))

        if conversation_log:
            story.append(PageBreak())

        for message in conversation_log:
            display_role = role_display_name(
                message["role"]
            )

            timestamp = message["time"].strftime(
                "%H:%M:%S"
            )

            heading_text = (
                f"{display_role} - {timestamp}"
            )

            formatted_message, has_arabic = prepare_pdf_text(
                message["content"]
            )

            story.append(
                Paragraph(
                    html.escape(heading_text),
                    (
                        role_style_right
                        if has_arabic
                        else role_style_left
                    )
                )
            )

            story.append(
                Paragraph(
                    formatted_message,
                    (
                        message_style_right
                        if has_arabic
                        else message_style_left
                    )
                )
            )

        def add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont("ChatUnicode", 8)

            page_text = f"Page {doc.page}"

            canvas.drawRightString(
                A4[0] - 18 * mm,
                10 * mm,
                page_text
            )

            canvas.restoreState()

        document.build(
            story,
            onFirstPage=add_page_number,
            onLaterPages=add_page_number
        )

        return True

    except Exception as error:
        print("\nCould not generate the PDF file.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")
        return False


# =========================================================
# Export controls
# =========================================================

def choose_export_mode():
    print("\nChoose how the conversation should be saved:")
    print("1. Automatically save as TXT")
    print("2. Automatically save as PDF")
    print("3. Automatically save as both TXT and PDF")
    print("4. Manual saving only")

    choice = input(
        "\nChoose option 1 to 4: "
    ).strip()

    options = {
        "1": "txt",
        "2": "pdf",
        "3": "both",
        "4": "manual"
    }

    if choice not in options:
        print("Invalid choice. Using manual saving.")
        return "manual"

    return options[choice]


def save_chat(save_format, show_message=True):
    """
    Save the chat in the requested format.

    save_format:
    - txt
    - pdf
    - both
    """

    txt_saved = False
    pdf_saved = False

    if save_format in ("txt", "both"):
        txt_saved = export_chat_to_txt()

    if save_format in ("pdf", "both"):
        pdf_saved = export_chat_to_pdf()

    if show_message:
        print()

        if txt_saved:
            print(f"TXT saved to:")
            print(txt_export_path)

        if pdf_saved:
            print(f"PDF saved to:")
            print(pdf_export_path)

        if not txt_saved and not pdf_saved:
            print("No files were saved.")

        print()

    return txt_saved or pdf_saved


def autosave_chat():
    """
    Save automatically based on the selected mode.
    """

    if export_mode == "txt":
        export_chat_to_txt()

    elif export_mode == "pdf":
        export_chat_to_pdf()

    elif export_mode == "both":
        export_chat_to_txt()
        export_chat_to_pdf()


def manual_save_menu():
    print("\nChoose a file format:")
    print("1. TXT")
    print("2. PDF")
    print("3. Both")

    choice = input(
        "Choose option 1 to 3: "
    ).strip()

    if choice == "1":
        save_chat("txt")

    elif choice == "2":
        save_chat("pdf")

    elif choice == "3":
        save_chat("both")

    else:
        print("Invalid choice.\n")


# =========================================================
# System instruction selection
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

    choice = input(
        "\nChoose option 1 to 8: "
    ).strip()

    instructions = {
        "1": (
            "You are a helpful general-purpose AI assistant. "
            "Answer accurately, clearly, and practically. "
            "Answer in the same language as the user."
        ),

        "2": (
            "You are an expert AI Engineering tutor. "
            "Teach clearly and step by step. "
            "Use simple examples before advanced explanations. "
            "Explain mistakes and how to correct them. "
            "Answer in the same language as the user."
        ),

        "3": (
            "You are an expert programming assistant. "
            "Help the user write, understand, debug, and improve code. "
            "Provide complete usable code when appropriate. "
            "Explain errors clearly and step by step. "
            "Answer in the same language as the user."
        ),

        "4": (
            "You are a friendly and supportive chat companion. "
            "Speak naturally, casually, and respectfully. "
            "Be warm and engaging. "
            "Answer in the same language as the user."
        ),

        "5": (
            "You are a study coach and teaching assistant. "
            "Help the user understand lessons, prepare for exams, "
            "organize study sessions, and practice difficult topics. "
            "Answer in the same language as the user."
        ),

        "6": (
            "You are a professional writing assistant. "
            "Help with writing, rewriting, summarizing, translating, "
            "proofreading, and improving professional text. "
            "Answer in the same language as the user."
        ),

        "7": (
            "You are a concise AI assistant. "
            "Give direct and accurate answers. "
            "Avoid unnecessary details unless requested. "
            "Answer in the same language as the user."
        )
    }

    if choice in instructions:
        return instructions[choice]

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

        custom_instruction = "\n".join(
            custom_lines
        ).strip()

        if custom_instruction:
            return custom_instruction

    print("Using the general-purpose instruction.")

    return instructions["1"]


# =========================================================
# Provider selection
# =========================================================

def choose_provider():
    print("Choose an AI provider:")
    print("1. Gemini API")
    print("2. LM Studio Local Server")

    while True:
        choice = input(
            "\nChoose option 1 or 2: "
        ).strip()

        if choice in ("1", "2"):
            return choice

        print("Invalid choice. Enter 1 or 2.")


# =========================================================
# Temperature selection
# =========================================================

def choose_temperature():
    if provider_choice == "1":
        default_temperature = 1.0
    else:
        default_temperature = 0.7

    print("\nTemperature controls randomness.")
    print(f"Recommended default: {default_temperature}")

    choice = input(
        f"Keep temperature at {default_temperature}? "
        "(yes/no): "
    ).strip().lower()

    if choice in ("yes", "y", ""):
        return default_temperature

    try:
        value = float(
            input("Enter temperature: ").strip()
        )

        if value < 0:
            raise ValueError

        return value

    except ValueError:
        print(
            f"Invalid value. Using {default_temperature}."
        )

        return default_temperature


# =========================================================
# Token selection
# =========================================================

def choose_max_tokens():
    print("\nChoose maximum output tokens:")
    print("1. Short answers: 700")
    print("2. Normal explanations: 1500")
    print("3. Coding and debugging: 2500")
    print("4. Deep explanations: 4000")
    print("5. Long writing: 3000")
    print("6. Custom value")

    choice = input(
        "\nChoose option 1 to 6: "
    ).strip()

    options = {
        "1": 700,
        "2": 1500,
        "3": 2500,
        "4": 4000,
        "5": 3000
    }

    if choice in options:
        return options[choice]

    if choice == "6":
        try:
            value = int(
                input(
                    "Enter maximum output tokens: "
                ).strip()
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
# Gemini setup and requests
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
        print("\ngoogle-genai is not installed.")
        print(
            "Run: py -m pip install --upgrade google-genai"
        )
        sys.exit(1)

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMINI_KEY")
    )

    if not api_key:
        print("\nGemini API key was not found.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

    print("\nChoose a Gemini model:")
    print("1. gemini-3.5-flash")
    print("2. Enter a custom model name")

    choice = input(
        "\nChoose option 1 or 2: "
    ).strip()

    if choice == "1":
        model_name = "gemini-3.5-flash"

    elif choice == "2":
        model_name = input(
            "Enter model name: "
        ).strip()

        if not model_name:
            model_name = "gemini-3.5-flash"

    else:
        model_name = "gemini-3.5-flash"

    try:
        gemini_chat = create_gemini_chat()

    except Exception as error:
        print("\nCould not create the Gemini chat.")
        print(error)
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

            if (
                error_code in retryable_codes
                and attempt < max_attempts
            ):
                wait_seconds = (
                    2 ** (attempt - 1)
                    + random.uniform(0, 1)
                )

                print(
                    f"\nTemporary API error {error_code}. "
                    f"Retrying in {wait_seconds:.1f} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            print("\nGemini request failed.")
            print(error)
            return None

        except KeyboardInterrupt:
            print("\nRequest cancelled.")
            return None

        except Exception as error:
            print("\nUnexpected Gemini error.")
            print(error)
            return None

    return None


# =========================================================
# LM Studio setup and requests
# =========================================================

def reset_lm_messages():
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
        print(
            "Run: py -m pip install --upgrade openai"
        )
        sys.exit(1)

    lm_client = OpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio"
    )

    try:
        model_response = lm_client.models.list()

    except APIConnectionError:
        print("\nCould not connect to LM Studio.")
        print("Start the server at:")
        print(LM_STUDIO_BASE_URL)
        sys.exit(1)

    model_ids = [
        model.id
        for model in model_response.data
    ]

    if not model_ids:
        print("\nNo LM Studio models were found.")
        sys.exit(1)

    print("\nAvailable LM Studio models:")

    for index, local_model in enumerate(
        model_ids,
        start=1
    ):
        print(f"{index}. {local_model}")

    choice = input(
        "\nChoose a model number: "
    ).strip()

    try:
        index = int(choice) - 1
        model_name = model_ids[index]

    except (ValueError, IndexError):
        model_name = model_ids[0]

        print(
            f"Using the first model: {model_name}"
        )

    reset_lm_messages()


def send_lm_studio_message(prompt, max_attempts=3):
    global lm_messages

    lm_messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

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
            lm_messages.pop()
            return None

        except APIStatusError as error:
            if (
                error.status_code
                in {408, 429, 500, 502, 503, 504}
                and attempt < max_attempts
            ):
                wait_seconds = 2 ** (attempt - 1)

                print(
                    f"\nTemporary LM Studio error. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            print("\nLM Studio request failed.")
            print(error)
            lm_messages.pop()
            return None

        except KeyboardInterrupt:
            print("\nRequest cancelled.")
            lm_messages.pop()
            return None

        except Exception as error:
            print("\nUnexpected LM Studio error.")
            print(error)
            lm_messages.pop()
            return None

    lm_messages.pop()
    return None


# =========================================================
# Unified AI request function
# =========================================================

def generate_ai_response(prompt):
    """
    Send a prompt to the selected AI provider and return
    only the generated response text.

    Returns:
        str: The AI response text.
        None: If the request fails.
    """

    global last_ai_response

    try:
        if provider_choice == "1":
            last_ai_response = send_gemini_message(prompt)

            if last_ai_response is None:
                return None

            answer = last_ai_response.text or ""

            if not answer:
                return "[Gemini returned no visible text.]"

            return answer

        last_ai_response = send_lm_studio_message(prompt)

        if last_ai_response is None:
            return None

        answer = (
            last_ai_response.choices[0].message.content
            or ""
        )

        if not answer:
            return "[Local model returned no visible text.]"

        return answer

    except Exception as error:
        last_ai_response = None

        print("\nAI request failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")

        return None


# =========================================================
# Start program
# =========================================================

provider_choice = choose_provider()

if provider_choice == "1":
    provider_name = "Gemini API"
else:
    provider_name = "LM Studio Local Server"

system_instruction = choose_system_instruction()
temperature_value = choose_temperature()
max_tokens_value = choose_max_tokens()

if provider_choice == "1":
    setup_gemini()
else:
    setup_lm_studio()

export_mode = choose_export_mode()

record_message(
    "system",
    (
        f"Chat started using {provider_name}, "
        f"model {model_name}."
    )
)

autosave_chat()


# =========================================================
# Chat information
# =========================================================

print("\n====================================")
print("Chat started")
print("====================================")
print(f"Provider: {provider_name}")
print(f"Model: {model_name}")
print(f"Export folder: {export_directory}")

print("\nCommands:")
print("clear       Clear the model's conversation memory")
print("system      Change the system instruction")
print("instruction Show the current system instruction")
print("save        Open the save menu")
print("save txt    Save as TXT")
print("save pdf    Save as PDF")
print("save both   Save as TXT and PDF")
print("exit        Save and close the program")
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
            record_message(
                "system",
                "Chat session ended."
            )

            autosave_chat()

            print("\nChat ended.")

            if export_mode != "manual":
                print("Latest conversation files:")
                
                if export_mode in ("txt", "both"):
                    print(txt_export_path)

                if export_mode in ("pdf", "both"):
                    print(pdf_export_path)

            break

        if command == "save":
            manual_save_menu()
            continue

        if command == "save txt":
            save_chat("txt")
            continue

        if command == "save pdf":
            save_chat("pdf")
            continue

        if command == "save both":
            save_chat("both")
            continue

        if command == "instruction":
            print("\nCurrent system instruction:")
            print(system_instruction)
            print()
            continue

        if command == "clear":
            if provider_choice == "1":
                gemini_chat = create_gemini_chat()
            else:
                reset_lm_messages()

            record_message(
                "system",
                "The model's conversation memory was cleared."
            )

            autosave_chat()

            print(
                "Conversation memory cleared. "
                "The exported transcript was not deleted.\n"
            )

            continue

        if command == "system":
            system_instruction = (
                choose_system_instruction()
            )

            if provider_choice == "1":
                gemini_chat = create_gemini_chat()
            else:
                reset_lm_messages()

            record_message(
                "system",
                (
                    "System instruction changed to:\n"
                    + system_instruction
                )
            )

            autosave_chat()

            print(
                "\nSystem instruction changed. "
                "Model memory was cleared.\n"
            )

            continue

        # Store the user message before sending it.
        record_message("user", prompt)

        start_time = time.perf_counter()

        answer = generate_ai_response(prompt)

        time_passed = (
            time.perf_counter() - start_time
        )

        if answer is None:
            if provider_choice == "1":
                failure_message = (
                    "The Gemini request failed. "
                    "No response received."
                )
            else:
                failure_message = (
                    "The LM Studio request failed. "
                    "No response received."
                )

            record_message(
                "system",
                failure_message
            )

            autosave_chat()

            print("\nYou can enter another prompt.\n")
            continue

        if provider_choice == "1":
            print("\nGemini answer:\n")
        else:
            print("\nLocal LLM answer:\n")

        print(answer)

        record_message("assistant", answer)
        autosave_chat()

        print("\n--- Response information ---")
        print(f"Time passed: {time_passed:.2f} Sec")

        if provider_choice == "1":
            usage = last_ai_response.usage_metadata

            if usage:
                print(
                    f"Input tokens: "
                    f"{usage.prompt_token_count}"
                )

                print(
                    f"Output tokens: "
                    f"{usage.candidates_token_count}"
                )

                print(
                    f"Total tokens: "
                    f"{usage.total_token_count}"
                )

        else:
            if last_ai_response.usage:
                print(
                    f"Input tokens: "
                    f"{last_ai_response.usage.prompt_tokens}"
                )

                print(
                    f"Output tokens: "
                    f"{last_ai_response.usage.completion_tokens}"
                )

                print(
                    f"Total tokens: "
                    f"{last_ai_response.usage.total_tokens}"
                )

        print()


except KeyboardInterrupt:
    record_message(
        "system",
        "Chat ended using Ctrl+C."
    )

    autosave_chat()
    print("\nChat saved and closed.")


finally:
    try:
        if gemini_client is not None:
            gemini_client.close()

        if lm_client is not None:
            lm_client.close()

    except Exception:
        pass