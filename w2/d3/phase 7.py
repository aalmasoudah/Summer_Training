import os
import random
import time

import fitz
import streamlit as st


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
# Existing AI configuration
# =========================================================

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

provider_choice = None
provider_name = None
model_name = None

system_instruction = (
    "You are an expert AI Engineering tutor. "
    "Teach clearly and step by step. "
    "Use simple examples before advanced explanations. "
    "Explain mistakes and how to correct them. "
    "Answer in the same language as the user."
)

temperature_value = 1.0
max_tokens_value = 1500

gemini_client = None
lm_client = None
last_ai_response = None

# Prevent extremely large PDF prompts.
MAX_PDF_CHARACTERS = 30000

SUBJECT_INSTRUCTIONS = {
    "Computer Science": (
        "You are a Computer Science tutor. Explain programming and "
        "computing concepts clearly, using practical examples when helpful."
    ),
    "Mathematics": (
        "You are a Mathematics tutor. Explain each solution clearly and "
        "show the important reasoning and calculation steps."
    ),
    "General Science": (
        "You are a General Science tutor. Explain scientific concepts "
        "accurately using simple language and relevant examples."
    )
}

STUDY_MODE_INSTRUCTIONS = {
    "Ask a Question": (
        "Answer the student's question using the uploaded PDF as the main "
        "source. If the answer is not available in the PDF, clearly say so."
    ),
    "Explain a Topic": (
        "Explain the requested topic simply using the uploaded PDF. "
        "Include one clear example and end with a short summary."
    ),
    "Generate a Quiz": (
        "Generate exactly five multiple-choice questions using the uploaded "
        "PDF. Each question must include exactly four options labeled A, B, "
        "C, and D. After each question, include the correct answer and a "
        "short explanation."
    )
}


# =========================================================
# Gemini setup and request
# =========================================================

def setup_gemini():
    """Create the Gemini client using the existing API-key setup."""

    global gemini_client
    global model_name

    if genai is None:
        raise RuntimeError(
            "google-genai is not installed. Run: "
            "py -m pip install --upgrade google-genai"
        )

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMINI_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Gemini API key was not found. Set GEMINI_API_KEY "
            "or GEMINI_KEY in your environment variables."
        )

    model_name = "gemini-3.5-flash"
    gemini_client = genai.Client(api_key=api_key)


def send_gemini_message(prompt, max_attempts=5):
    """Send one independent question to Gemini."""

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
            chat = gemini_client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    temperature=temperature_value,
                    max_output_tokens=max_tokens_value,
                    system_instruction=system_instruction
                )
            )

            return chat.send_message(prompt)

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

                time.sleep(wait_seconds)
                continue

            raise RuntimeError(
                f"Gemini request failed: {error}"
            ) from error

        except Exception as error:
            raise RuntimeError(
                f"Unexpected Gemini error: {error}"
            ) from error

    return None


# =========================================================
# LM Studio setup and request
# =========================================================

def setup_lm_studio():
    """Create the LM Studio client using the existing local URL."""

    global lm_client
    global model_name

    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed. Run: "
            "py -m pip install --upgrade openai"
        )

    lm_client = OpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio"
    )

    try:
        model_response = lm_client.models.list()

    except APIConnectionError as error:
        raise RuntimeError(
            "Could not connect to LM Studio. Start the local "
            f"server at {LM_STUDIO_BASE_URL}."
        ) from error

    model_ids = [
        model.id
        for model in model_response.data
    ]

    if not model_ids:
        raise RuntimeError(
            "No loaded LM Studio models were found."
        )

    model_name = model_ids[0]


def send_lm_studio_message(prompt, max_attempts=3):
    """Send one independent question to the LM Studio model."""

    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            return lm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature_value,
                max_tokens=max_tokens_value
            )

        except APIConnectionError as error:
            raise RuntimeError(
                "Connection to LM Studio was lost."
            ) from error

        except APIStatusError as error:
            if (
                error.status_code
                in {408, 429, 500, 502, 503, 504}
                and attempt < max_attempts
            ):
                wait_seconds = 2 ** (attempt - 1)
                time.sleep(wait_seconds)
                continue

            raise RuntimeError(
                f"LM Studio request failed: {error}"
            ) from error

        except Exception as error:
            raise RuntimeError(
                f"Unexpected LM Studio error: {error}"
            ) from error

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
        st.error(str(error))
        return None


# =========================================================
# PDF text extraction
# =========================================================

def extract_pdf_text(uploaded_pdf):
    """Extract and combine readable text from every PDF page."""

    try:
        pdf_bytes = uploaded_pdf.getvalue()
        page_texts = []

        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as document:

            for page in document:
                page_texts.append(page.get_text())

        return "\n".join(page_texts).strip()

    except Exception as error:
        raise RuntimeError(
            f"Could not read the uploaded PDF: {error}"
        ) from error


def build_pdf_question_prompt(
    pdf_text,
    student_input,
    subject_instruction,
    mode_instruction,
    selected_mode
):
    """Build one PDF-grounded prompt for the selected study mode."""

    limited_pdf_text = pdf_text[:MAX_PDF_CHARACTERS]

    return f"""
SUBJECT TUTOR INSTRUCTION:
{subject_instruction}

SELECTED STUDY MODE:
{selected_mode}

MODE INSTRUCTION:
{mode_instruction}

General rules:
- Use the uploaded PDF as the main source.
- Do not invent information that is not supported by the PDF.
- If the requested information is not available in the PDF, clearly say:
  "The answer is not available in the uploaded PDF."
- Answer in the same language as the student's input.
- Follow the selected mode instructions exactly.

PDF TEXT:
--------------------
{limited_pdf_text}
--------------------

STUDENT INPUT:
{student_input}
""".strip()


# =========================================================
# Streamlit application
# =========================================================

st.set_page_config(
    page_title="AI Course Tutor",
    page_icon="🎓"
)

st.title("AI Course Tutor")

provider_label = st.sidebar.selectbox(
    "AI provider",
    (
        "Gemini API",
        "LM Studio Local Server"
    )
)

selected_subject = st.sidebar.selectbox(
    "Tutor subject",
    tuple(SUBJECT_INSTRUCTIONS.keys())
)

selected_subject_instruction = SUBJECT_INSTRUCTIONS[
    selected_subject
]

selected_mode = st.sidebar.selectbox(
    "Study mode",
    (
        "Ask a Question",
        "Explain a Topic",
        "Generate a Quiz"
    )
)

selected_mode_instruction = STUDY_MODE_INSTRUCTIONS[
    selected_mode
]

if provider_label == "Gemini API":
    provider_choice = "1"
    provider_name = "Gemini API"
    temperature_value = 1.0
else:
    provider_choice = "2"
    provider_name = "LM Studio Local Server"
    temperature_value = 0.7

try:
    if provider_choice == "1":
        setup_gemini()
    else:
        setup_lm_studio()

except RuntimeError as error:
    st.error(str(error))
    st.stop()


# =========================================================
# PDF uploader
# =========================================================

uploaded_pdf = st.file_uploader(
    "Upload a course PDF",
    type=["pdf"]
)

extracted_pdf_text = ""
pdf_ready = False

if uploaded_pdf is not None:
    st.write(f"PDF filename: {uploaded_pdf.name}")

    try:
        extracted_pdf_text = extract_pdf_text(
            uploaded_pdf
        )

        if not extracted_pdf_text:
            st.error(
                "No readable text was found in this PDF. "
                "It may contain scanned images instead of "
                "selectable text."
            )

        else:
            pdf_ready = True

            st.write(
                "Extracted characters: "
                f"{len(extracted_pdf_text)}"
            )

            characters_to_send = min(
                len(extracted_pdf_text),
                MAX_PDF_CHARACTERS
            )

            st.write(
                "Characters available to the AI: "
                f"{characters_to_send}"
            )

            if len(extracted_pdf_text) > MAX_PDF_CHARACTERS:
                st.warning(
                    "The PDF is large. Only the first "
                    f"{MAX_PDF_CHARACTERS} characters will be "
                    "sent with the question."
                )

    except RuntimeError as error:
        st.error(str(error))
else:
    st.info(
        "Upload a readable PDF before asking a "
        "course-material question."
    )


# =========================================================
# Question and PDF-grounded response feature
# =========================================================

if selected_mode == "Ask a Question":
    input_label = "Enter your question"
    input_placeholder = (
        "Ask a question about the uploaded course PDF..."
    )

elif selected_mode == "Explain a Topic":
    input_label = "Enter the topic to explain"
    input_placeholder = (
        "Example: Explain recursion simply..."
    )

else:
    input_label = "Enter the quiz topic"
    input_placeholder = (
        "Example: Generate a quiz about Chapter 2..."
    )

question = st.text_area(
    input_label,
    placeholder=input_placeholder
)

send_button = st.button(
    "Send",
    disabled=not pdf_ready
)

if send_button:
    if not question.strip():
        st.warning(
            "Please enter your study request before pressing Send."
        )

    else:
        combined_prompt = build_pdf_question_prompt(
            extracted_pdf_text,
            question.strip(),
            selected_subject_instruction,
            selected_mode_instruction,
            selected_mode
        )

        with st.spinner("Generating response from the PDF..."):
            ai_response = generate_ai_response(
                combined_prompt
            )

        if ai_response is not None:
            st.subheader("AI Response")
            st.write(ai_response)
