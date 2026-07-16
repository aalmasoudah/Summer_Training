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
max_tokens_value = 5000

gemini_client = None
lm_client = None

# Safety limits for uploads and prompts.
MAX_PDF_SIZE_MB = 10
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
MAX_PDF_CHARACTERS = 25000

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

    local_system_instruction = (
        system_instruction
        + "\n/no_think"
        + "\nReturn only the final answer. Do not spend the response "
        "budget on hidden reasoning."
    )

    local_prompt = (
        prompt
        + "\n\n/no_think"
        + "\nProvide the final answer directly and clearly."
    )

    messages = [
        {
            "role": "system",
            "content": local_system_instruction
        },
        {
            "role": "user",
            "content": local_prompt
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
    Send a prompt to the selected AI provider, store the current
    response in Streamlit session state, and return its text.
    """

    try:
        if provider_choice == "1":
            response = send_gemini_message(prompt)

            if response is None:
                st.session_state.current_ai_response = None
                return None

            answer = response.text or ""

            if not answer:
                answer = "[Gemini returned no visible text.]"

        else:
            response = send_lm_studio_message(prompt)

            if response is None:
                st.session_state.current_ai_response = None
                return None

            if not response.choices:
                raise RuntimeError(
                    "The local model returned no response choices."
                )

            choice = response.choices[0]
            message = choice.message

            answer = (
                getattr(message, "content", None)
                or ""
            ).strip()

            if not answer:
                reasoning = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                    or ""
                ).strip()

                finish_reason = (
                    getattr(choice, "finish_reason", None)
                    or "unknown"
                )

                if reasoning and finish_reason == "length":
                    raise RuntimeError(
                        "The local model used the full output-token "
                        "allowance for reasoning before producing a final "
                        "answer. Non-thinking mode was requested, but the "
                        "model still reached the limit. Confirm that the "
                        "loaded model supports /no_think, or use a smaller "
                        "prompt."
                    )

                if reasoning:
                    raise RuntimeError(
                        "The local model produced reasoning but no visible "
                        "final answer. Confirm that non-thinking mode is "
                        "enabled for the loaded Qwen model."
                    )

                raise RuntimeError(
                    "The local model returned an empty final answer. "
                    f"Finish reason: {finish_reason}."
                )

        st.session_state.current_ai_response = answer
        return answer

    except Exception as error:
        st.session_state.current_ai_response = None
        st.error(str(error))
        return None


# =========================================================
# PDF text extraction
# =========================================================

def extract_pdf_text(uploaded_pdf):
    """Validate a PDF and extract readable text from every page."""

    if uploaded_pdf is None:
        raise RuntimeError(
            "No PDF file was uploaded."
        )

    filename = uploaded_pdf.name or ""

    if not filename.lower().endswith(".pdf"):
        raise RuntimeError(
            "Only PDF files are accepted."
        )

    mime_type = getattr(uploaded_pdf, "type", "")

    if mime_type not in ("application/pdf", ""):
        raise RuntimeError(
            "The selected file is not recognized as a PDF."
        )

    pdf_bytes = uploaded_pdf.getvalue()

    if not pdf_bytes:
        raise RuntimeError(
            "The uploaded PDF is empty."
        )

    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise RuntimeError(
            f"The PDF is too large. Maximum size is "
            f"{MAX_PDF_SIZE_MB} MB."
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError(
            "The selected file does not appear to be a valid PDF."
        )

    try:
        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as document:

            if document.page_count == 0:
                raise RuntimeError(
                    "The uploaded PDF contains no pages."
                )

            page_texts = []

            for page in document:
                page_texts.append(
                    page.get_text("text")
                )

    except RuntimeError:
        raise

    except (fitz.FileDataError, ValueError) as error:
        raise RuntimeError(
            "The PDF could not be opened. It may be damaged, "
            "password-protected, or not a valid PDF."
        ) from error

    except Exception as error:
        raise RuntimeError(
            "The PDF could not be processed. Please try a "
            "different readable PDF."
        ) from error

    combined_text = "\n".join(page_texts).strip()

    if not combined_text:
        raise RuntimeError(
            "No readable text was found in the PDF. It may contain "
            "scanned images instead of selectable text."
        )

    return combined_text


def build_pdf_question_prompt(
    pdf_text,
    student_input,
    subject_instruction,
    mode_instruction,
    selected_mode
):
    """Build one PDF-grounded prompt for the selected study mode."""

    if not pdf_text or not pdf_text.strip():
        raise RuntimeError(
            "No readable PDF text is available for this request."
        )

    if not student_input or not student_input.strip():
        raise RuntimeError(
            "Please enter your study request."
        )

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

    if not pdf_text or not pdf_text.strip():
        raise RuntimeError(
            "No readable PDF text is available for the quiz."
        )

    if not quiz_topic or not quiz_topic.strip():
        raise RuntimeError(
            "Please enter a quiz topic."
        )

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
            "The quiz response was not formatted correctly. "
            "Please generate the question again."
        )

    json_text = cleaned_response[
        json_start:json_end + 1
    ]

    try:
        quiz_data = json.loads(json_text)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The quiz response was not formatted correctly. "
            "Please generate the question again."
        ) from error

    if not isinstance(quiz_data, dict):
        raise RuntimeError(
            "The AI quiz response was not in the expected object format. "
            "Please generate the question again."
        )

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


def initialize_session_state():
    """Initialize only the application data that must survive reruns."""

    defaults = {
        "extracted_pdf_text": "",
        "uploaded_pdf_filename": None,
        "current_ai_response": None,
        "current_quiz_question": None,
        "correct_quiz_answer": None,
        "quiz_explanation": None,
        "answer_submitted": False
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_quiz_state():
    """Clear the current quiz without storing history or scores."""

    st.session_state.current_quiz_question = None
    st.session_state.correct_quiz_answer = None
    st.session_state.quiz_explanation = None
    st.session_state.answer_submitted = False


def clear_generated_content():
    """Clear generated output when the provider, subject, or mode changes."""

    st.session_state.current_ai_response = None
    clear_quiz_state()


def clear_document_state():
    """Clear the uploaded document and all generated content."""

    st.session_state.extracted_pdf_text = ""
    st.session_state.uploaded_pdf_filename = None
    clear_generated_content()


def generate_quiz_question(
    pdf_text,
    quiz_topic,
    subject_instruction,
    previous_question=""
):
    """Generate and store exactly one validated quiz question."""

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

    # Keep the options with the current question so no separate
    # quiz-options state field is needed.
    st.session_state.current_quiz_question = {
        "text": quiz_data["question"],
        "options": quiz_data["options"]
    }
    st.session_state.correct_quiz_answer = (
        quiz_data["correct_answer"]
    )
    st.session_state.quiz_explanation = (
        quiz_data["explanation"]
    )
    st.session_state.answer_submitted = False

    return True


# =========================================================
# Streamlit application
# =========================================================

st.set_page_config(
    page_title="AI Course Tutor",
    page_icon="🎓"
)

st.title("AI Course Tutor")

initialize_session_state()

provider_label = st.sidebar.selectbox(
    "AI provider",
    (
        "Gemini API",
        "LM Studio Local Server"
    ),
    on_change=clear_generated_content
)

selected_subject = st.sidebar.selectbox(
    "Tutor subject",
    tuple(SUBJECT_INSTRUCTIONS.keys()),
    on_change=clear_generated_content
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
    ),
    on_change=clear_generated_content
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
        st.sidebar.caption(
            f"Loaded local model: {model_name}"
        )

except RuntimeError as error:
    st.error(str(error))
    st.stop()


# =========================================================
# PDF uploader
# =========================================================

uploaded_pdf = st.file_uploader(
    "Upload a course PDF",
    type=["pdf"],
    help=f"PDF files only. Maximum size: {MAX_PDF_SIZE_MB} MB."
)

pdf_ready = False

if uploaded_pdf is not None:
    uploaded_filename = uploaded_pdf.name
    filename_changed = (
        st.session_state.uploaded_pdf_filename
        != uploaded_filename
    )

    if filename_changed:
        clear_generated_content()

    try:
        if (
            filename_changed
            or not st.session_state.extracted_pdf_text
        ):
            extracted_text = extract_pdf_text(
                uploaded_pdf
            )

            st.session_state.uploaded_pdf_filename = (
                uploaded_filename
            )
            st.session_state.extracted_pdf_text = (
                extracted_text
            )

        extracted_pdf_text = (
            st.session_state.extracted_pdf_text
        )
        pdf_ready = True

        st.write(
            "PDF filename: "
            f"{st.session_state.uploaded_pdf_filename}"
        )

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
                "included in each AI prompt."
            )

    except RuntimeError as error:
        st.session_state.extracted_pdf_text = ""
        st.session_state.uploaded_pdf_filename = (
            uploaded_filename
        )
        st.error(str(error))

    except Exception:
        st.session_state.extracted_pdf_text = ""
        st.session_state.uploaded_pdf_filename = (
            uploaded_filename
        )
        st.error(
            "The PDF could not be processed. Please choose "
            "another valid PDF file."
        )

else:
    if st.session_state.uploaded_pdf_filename is not None:
        clear_document_state()

    extracted_pdf_text = ""

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
            try:
                combined_prompt = build_pdf_question_prompt(
                    st.session_state.extracted_pdf_text,
                    question.strip(),
                    selected_subject_instruction,
                    selected_mode_instruction,
                    selected_mode
                )

                with st.spinner(
                    "Generating response from the PDF..."
                ):
                    generate_ai_response(
                        combined_prompt
                    )

            except RuntimeError as error:
                st.error(str(error))

            except Exception:
                st.error(
                    "The request could not be prepared. "
                    "Please try again."
                )

    if st.session_state.current_ai_response:
        st.subheader("AI Response")
        st.write(
            st.session_state.current_ai_response
        )


# =========================================================
# Interactive quiz mode
# =========================================================

else:
    quiz_topic_input = st.text_area(
        "Enter the quiz topic",
        placeholder="Example: Chapter 2 or Python functions..."
    )

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
                        st.session_state.extracted_pdf_text,
                        quiz_topic,
                        selected_subject_instruction
                    )

            except RuntimeError as error:
                st.error(str(error))

            except Exception:
                st.error(
                    "The quiz could not be generated. "
                    "Please try again."
                )

    quiz_data = st.session_state.current_quiz_question

    if quiz_data:
        st.subheader("Quiz Question")
        st.write(quiz_data["text"])

        # The widget manages its own selected value. Application data is
        # limited to the requested session-state fields.
        radio_key = (
            "quiz_answer_"
            f"{abs(hash(quiz_data['text']))}"
        )

        selected_answer = st.radio(
            "Choose one answer:",
            ("A", "B", "C", "D"),
            format_func=lambda label: (
                f"{label}. "
                f"{quiz_data['options'][label]}"
            ),
            index=None,
            key=radio_key,
            disabled=st.session_state.answer_submitted
        )

        submit_answer = st.button(
            "Submit Answer",
            disabled=st.session_state.answer_submitted
        )

        if submit_answer:
            if selected_answer is None:
                st.warning(
                    "Please choose an answer before submitting."
                )

            else:
                st.session_state.answer_submitted = True

        if st.session_state.answer_submitted:
            correct_answer = (
                st.session_state.correct_quiz_answer
            )

            if selected_answer == correct_answer:
                st.success("Correct!")
            else:
                st.error(
                    "Incorrect. The correct answer is "
                    f"{correct_answer}."
                )

            st.write("**Explanation:**")
            st.write(
                st.session_state.quiz_explanation
            )

            next_question = st.button("Next Question")

            if next_question:
                quiz_topic = quiz_topic_input.strip()

                if not quiz_topic:
                    st.warning(
                        "Keep or enter a quiz topic before "
                        "generating the next question."
                    )

                else:
                    previous_question = quiz_data["text"]

                    try:
                        with st.spinner(
                            "Generating the next question..."
                        ):
                            question_created = (
                                generate_quiz_question(
                                    st.session_state.extracted_pdf_text,
                                    quiz_topic,
                                    selected_subject_instruction,
                                    previous_question
                                )
                            )

                        if question_created:
                            st.rerun()

                    except RuntimeError as error:
                        st.error(str(error))

                    except Exception:
                        st.error(
                            "The next quiz question could not be generated. "
                            "Please try again."
                        )
