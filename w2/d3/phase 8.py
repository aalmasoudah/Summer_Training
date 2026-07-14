import json
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
        "Generate exactly one multiple-choice question using the uploaded "
        "PDF. Include exactly four options labeled A, B, C, and D, one "
        "correct answer, and a short explanation."
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
# Quiz generation and parsing
# =========================================================

def build_quiz_prompt(
    pdf_text,
    quiz_topic,
    subject_instruction,
    previous_question=""
):
    """Build a strict one-question quiz prompt."""

    limited_pdf_text = pdf_text[:MAX_PDF_CHARACTERS]

    previous_question_rule = ""

    if previous_question:
        previous_question_rule = (
            "\n- Create a different question from this previous one:\n"
            f'  "{previous_question}"'
        )

    return f"""
SUBJECT TUTOR INSTRUCTION:
{subject_instruction}

TASK:
Generate exactly one multiple-choice question about the student's requested
topic using the uploaded PDF as the main source.

STRICT OUTPUT FORMAT:
Return only one valid JSON object. Do not use Markdown or code fences.

{{
  "question": "Question text",
  "options": {{
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct_answer": "A",
  "explanation": "Short explanation of why the answer is correct"
}}

RULES:
- Generate exactly one question.
- Include exactly four options with the keys A, B, C, and D.
- Set correct_answer to only A, B, C, or D.
- Keep the explanation short and clear.
- Use the uploaded PDF as the main source.
- Do not invent facts that are unsupported by the PDF.
- If the topic is unavailable in the PDF, make the question test that fact
  and explain that the requested material was not found.
- Use the same language as the student's topic.{previous_question_rule}

PDF TEXT:
--------------------
{limited_pdf_text}
--------------------

QUIZ TOPIC:
{quiz_topic}
""".strip()


def parse_quiz_response(response_text):
    """Parse and validate the model's predictable JSON quiz response."""

    if not response_text:
        raise RuntimeError(
            "The AI returned an empty quiz response."
        )

    cleaned_response = response_text.strip()

    json_start = cleaned_response.find("{")
    json_end = cleaned_response.rfind("}")

    if json_start == -1 or json_end == -1:
        raise RuntimeError(
            "The AI did not return the required JSON quiz format."
        )

    json_text = cleaned_response[
        json_start:json_end + 1
    ]

    try:
        quiz_data = json.loads(json_text)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The AI returned invalid JSON for the quiz."
        ) from error

    question = quiz_data.get("question")
    options = quiz_data.get("options")
    correct_answer = str(
        quiz_data.get("correct_answer", "")
    ).strip().upper()
    explanation = quiz_data.get("explanation")

    if not isinstance(question, str) or not question.strip():
        raise RuntimeError(
            "The generated quiz is missing its question."
        )

    if not isinstance(options, dict):
        raise RuntimeError(
            "The generated quiz options are not valid."
        )

    required_labels = {"A", "B", "C", "D"}

    if set(options.keys()) != required_labels:
        raise RuntimeError(
            "The generated quiz must contain only options A, B, C, and D."
        )

    for label in ("A", "B", "C", "D"):
        if (
            not isinstance(options[label], str)
            or not options[label].strip()
        ):
            raise RuntimeError(
                f"Quiz option {label} is empty or invalid."
            )

    if correct_answer not in required_labels:
        raise RuntimeError(
            "The quiz correct answer must be A, B, C, or D."
        )

    if (
        not isinstance(explanation, str)
        or not explanation.strip()
    ):
        raise RuntimeError(
            "The generated quiz is missing its explanation."
        )

    return {
        "question": question.strip(),
        "options": {
            label: options[label].strip()
            for label in ("A", "B", "C", "D")
        },
        "correct_answer": correct_answer,
        "explanation": explanation.strip()
    }


def initialize_quiz_state():
    """Create the quiz session-state fields once."""

    defaults = {
        "quiz_question": None,
        "quiz_options": None,
        "quiz_correct_answer": None,
        "quiz_explanation": None,
        "quiz_submitted": False,
        "quiz_selected_answer": None,
        "quiz_topic": "",
        "quiz_context": None,
        "quiz_radio_version": 0
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_quiz_state(keep_topic=False):
    """Clear the current quiz without storing scores or history."""

    saved_topic = (
        st.session_state.get("quiz_topic", "")
        if keep_topic
        else ""
    )

    st.session_state.quiz_question = None
    st.session_state.quiz_options = None
    st.session_state.quiz_correct_answer = None
    st.session_state.quiz_explanation = None
    st.session_state.quiz_submitted = False
    st.session_state.quiz_selected_answer = None
    st.session_state.quiz_topic = saved_topic
    st.session_state.quiz_context = None
    st.session_state.quiz_radio_version += 1


def generate_quiz_question(
    pdf_text,
    quiz_topic,
    subject_instruction,
    quiz_context,
    previous_question=""
):
    """Generate, validate, and store one quiz question."""

    quiz_prompt = build_quiz_prompt(
        pdf_text,
        quiz_topic,
        subject_instruction,
        previous_question
    )

    response_text = generate_ai_response(quiz_prompt)

    if response_text is None:
        return False

    quiz_data = parse_quiz_response(response_text)

    st.session_state.quiz_question = quiz_data["question"]
    st.session_state.quiz_options = quiz_data["options"]
    st.session_state.quiz_correct_answer = (
        quiz_data["correct_answer"]
    )
    st.session_state.quiz_explanation = (
        quiz_data["explanation"]
    )
    st.session_state.quiz_submitted = False
    st.session_state.quiz_selected_answer = None
    st.session_state.quiz_topic = quiz_topic
    st.session_state.quiz_context = quiz_context
    st.session_state.quiz_radio_version += 1

    return True


# =========================================================
# Streamlit application
# =========================================================

st.set_page_config(
    page_title="AI Course Tutor",
    page_icon="🎓"
)

st.title("AI Course Tutor")

initialize_quiz_state()

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
# Question, explanation, and interactive quiz features
# =========================================================

if selected_mode in ("Ask a Question", "Explain a Topic"):
    if selected_mode == "Ask a Question":
        input_label = "Enter your question"
        input_placeholder = (
            "Ask a question about the uploaded course PDF..."
        )

    else:
        input_label = "Enter the topic to explain"
        input_placeholder = (
            "Example: Explain recursion simply..."
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

            with st.spinner(
                "Generating response from the PDF..."
            ):
                ai_response = generate_ai_response(
                    combined_prompt
                )

            if ai_response is not None:
                st.subheader("AI Response")
                st.write(ai_response)


# =========================================================
# Interactive quiz mode
# =========================================================

else:
    quiz_topic_input = st.text_area(
        "Enter the quiz topic",
        placeholder="Example: Chapter 2 or Python functions...",
        key="quiz_topic_input"
    )

    current_quiz_context = None

    if pdf_ready:
        current_quiz_context = (
            f"{uploaded_pdf.name}|"
            f"{len(extracted_pdf_text)}|"
            f"{selected_subject}"
        )

    stored_context = st.session_state.quiz_context

    if (
        stored_context is not None
        and current_quiz_context != stored_context
    ):
        clear_quiz_state(keep_topic=False)

    generate_button = st.button(
        "Generate Question",
        disabled=not pdf_ready
    )

    if generate_button:
        quiz_topic = quiz_topic_input.strip()

        if not quiz_topic:
            st.warning(
                "Please enter a quiz topic before generating a question."
            )

        else:
            try:
                with st.spinner(
                    "Generating one quiz question..."
                ):
                    generate_quiz_question(
                        extracted_pdf_text,
                        quiz_topic,
                        selected_subject_instruction,
                        current_quiz_context
                    )

            except RuntimeError as error:
                st.error(str(error))

    if st.session_state.quiz_question:
        st.subheader("Quiz Question")
        st.write(st.session_state.quiz_question)

        radio_key = (
            "quiz_answer_"
            f"{st.session_state.quiz_radio_version}"
        )

        selected_answer = st.radio(
            "Choose one answer:",
            ("A", "B", "C", "D"),
            format_func=lambda label: (
                f"{label}. "
                f"{st.session_state.quiz_options[label]}"
            ),
            index=None,
            key=radio_key,
            disabled=st.session_state.quiz_submitted
        )

        submit_answer = st.button(
            "Submit Answer",
            disabled=st.session_state.quiz_submitted
        )

        if submit_answer:
            if selected_answer is None:
                st.warning(
                    "Please choose an answer before submitting."
                )

            else:
                st.session_state.quiz_selected_answer = (
                    selected_answer
                )
                st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            chosen_answer = (
                st.session_state.quiz_selected_answer
            )
            correct_answer = (
                st.session_state.quiz_correct_answer
            )

            if chosen_answer == correct_answer:
                st.success("Correct!")
            else:
                st.error(
                    "Incorrect. The correct answer is "
                    f"{correct_answer}."
                )

            st.write("**Explanation:**")
            st.write(st.session_state.quiz_explanation)

            next_question = st.button("Next Question")

            if next_question:
                previous_question = (
                    st.session_state.quiz_question
                )
                saved_quiz_topic = (
                    st.session_state.quiz_topic
                )

                try:
                    with st.spinner(
                        "Generating the next question..."
                    ):
                        question_created = (
                            generate_quiz_question(
                                extracted_pdf_text,
                                saved_quiz_topic,
                                selected_subject_instruction,
                                current_quiz_context,
                                previous_question
                            )
                        )

                    if question_created:
                        st.rerun()

                except RuntimeError as error:
                    st.error(str(error))
