from deep_translator import GoogleTranslator
def translation(query,language):
    translated = GoogleTranslator(source='auto', target=language).translate(query)
    return translated
print(translation("h","te"))