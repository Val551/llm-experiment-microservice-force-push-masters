import ollama

MODEL_NAME = "llama3.1:8b"

TRANSLATION_CONTEXT = """
You are a translation system.

Task:
Translate the input text into fluent English.

Rules:
- Output ONLY the English translation.
- Do not explain.
- Do not identify the language.
- Do not add notes or quotation marks.
- Preserve meaning as closely as possible.
- If the input is already English, return it unchanged.
- If the input is gibberish, malformed, or unintelligible, return it unchanged.
"""

CLASSIFICATION_CONTEXT = """
You are a language classifier.

Task:
Detect the language of the input text.

Rules:
- Output ONLY the English name of the language.
- Do not explain.
- Do not add extra words or punctuation.
- If the text is mostly English, output English.
- If the text is gibberish, malformed, or unintelligible, output Unknown.
"""


def get_language(post: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CLASSIFICATION_CONTEXT},
            {"role": "user", "content": post}
        ]
    )
    return response["message"]["content"].strip()


def get_translation(post: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TRANSLATION_CONTEXT},
            {"role": "user", "content": post}
        ]
    )
    return response["message"]["content"].strip()


def translate_content(content: str) -> tuple[bool, str]:
    try:
        language = get_language(content)

        if not isinstance(language, str):
            return (True, content)

        language = language.strip().lower()

        if language == "unknown":
            return (True, content)

        if language == "english":
            return (True, content)

        translation = get_translation(content)

        if not isinstance(translation, str) or translation.strip() == "":
            return (False, content)

        return (False, translation.strip())

    except Exception:
        return (True, content)
