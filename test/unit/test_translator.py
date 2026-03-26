from unittest.mock import patch
from src.translator import translate_content


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"


def test_llm_normal_response():
    with patch('src.translator.get_language', return_value="German"):
        with patch('src.translator.get_translation', return_value="This is a message in German"):
            is_english, translated_content = translate_content("Dies ist eine Nachricht auf Deutsch")
            assert is_english == False
            assert translated_content == "This is a message in German"

    with patch('src.translator.get_language', return_value="English"):
        is_english, translated_content = translate_content("Hello world")
        assert is_english == True
        assert translated_content == "Hello world"


def test_llm_gibberish_response():
    with patch('src.translator.get_language', return_value="Unknown"):
        is_english, translated_content = translate_content("asdkj123123")
        assert is_english == True
        assert translated_content == "asdkj123123"

    with patch('src.translator.get_language', side_effect=Exception("Ollama connection failed")):
        is_english, translated_content = translate_content("some text")
        assert is_english == True
        assert translated_content == "some text"

    with patch('src.translator.get_language', return_value="French"):
        with patch('src.translator.get_translation', return_value=""):
            is_english, translated_content = translate_content("Bonjour")
            assert is_english == False
            assert translated_content == "Bonjour"
