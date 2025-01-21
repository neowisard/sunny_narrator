import spacy

# analyze/module1.py
def analyze_text(text):
    nlp = spacy.load("ru_core_news_sm")

    doc = nlp(text)

    for ent in doc.ents:
        print(ent.text, ent.label_)

    # Ваш код анализа текста
    return result
