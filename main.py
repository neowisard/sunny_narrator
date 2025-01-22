from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from FB2 import FictionBook2, Author
from urllib import request
import dotenv
import yaml
import asyncio
# import libs.read.one as FBr
import libs.read.sec as fb2r
import libs.translate.llm as ta
from lxml.html.defs import table_tags
from sympy.physics.units import current

#from hypothesis import example

# Configs
config_dir = Path(__file__).parent.resolve() / "config"
# load yaml config
with open(config_dir / "config.yaml", 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)
# load .env config
config_env = dotenv.dotenv_values(config_dir / "config.env")

source_lang = config_env['SOURCE_LANG']
target_lang = config_env['TARGET_LANG']
country = config_env['COUNTRY']
max_len_chunk = int(config_env['MAX_LEN_CHUNK'])
max_len_paragraph = int(config_env['MAX_LEN_PARAGRAPH'])
fb2file = Path(__file__).parent.resolve() / config_env['FILE']
api_key = config_env['API_KEY']  # This is the default and can be omitted
base_url = config_env['API_BASE']



# Предполагается, что ta, source_lang, target_lang, orig_book и country уже определены

def translate_title():
    return ta.translate(source_lang, target_lang, orig_book.get_title(), country)

def translate_annotation():
    return ta.translate(source_lang, target_lang, orig_book.get_description(), country)

def translate_authors():
    authors = orig_book.get_authors()

    if len(authors) < 2:
        return None  # Используйте None вместо none

    for i in range(len(authors)):
        authors[i] = ta.translate(source_lang, target_lang, authors[i], country)

    return authors


def translate_tags():
    tags = orig_book.get_tags()

    # Check if tags is empty or has only one element
    if not tags or len(tags) < 2:
        return None  # Return None if there are no tags or only one tag

    # Translate each tag
    translated_tags = [
        ta.translate(source_lang, target_lang, tag, country)
        for tag in tags
    ]

    return translated_tags

def translate_series():
    series = orig_book.get_series()

    # Check if series is empty or has only one element
    if not series or len(series) < 2:
        return None  # Return None if there are no series or only one series

    # Translate each series element
    translated_series = [
        ta.translate(source_lang, target_lang, series_item, country)
        for series_item in series
    ]

    return translated_series

def print_book(book, indent=0):
    indent_str = ' ' * indent
    for attr, value in book.__dict__.items():
        if hasattr(value, '__dict__'):
            print(f"{indent_str}{attr}:")
            print_book(value, indent + 4)
        else:
            print(f"{indent_str}{attr}: {value}")


if __name__ == '__main__':
    # 1 Получить переменные и файл
    orig_book = fb2r.fb2book(fb2file)
    new_book = FictionBook2()

    # Открыть файл и Распарсить файл

    body, total_text_length = orig_book.get_body()
    chapters = orig_book.extract_chapters(body)





    # 2 Анализаторы
    # Make vocab by spaCy
    # Add user vocab
    # to make classification

    # 3 Переводчики (с передачей system\add promt
    # Translate meta


    with ThreadPoolExecutor(max_workers=5) as executor:
        future_title = executor.submit(translate_title)
        future_annotation = executor.submit(translate_annotation)
        future_authors = executor.submit(translate_authors)
        future_tags = executor.submit(translate_tags)
        future_series = executor.submit(translate_series)

        new_book.titleInfo.title = future_title.result()
        new_book.titleInfo.annotation = future_annotation.result()
        new_book.titleInfo.authors = future_authors.result()
        new_book.titleInfo.genres = future_tags.result()
        new_book.titleInfo.series = future_series.result()
        new_book.documentInfo.programUsed = "Sunny narrator ,github"
        new_book.titleInfo.translators = "Sunny narrator , automated AI translator"
        new_book.titleInfo.lang = target_lang
    #print_book(new_book)


    # Improve pictures

    # Chunking and translate body (stream)
    # Translating over 2-3 LLM and translator.


    sections = [
        {"title": "Введение", "para": ["Текст первого параграфа."]}
        ]

    #ta.translate(source_lang, target_lang, orig_book.get_description(), country)
    # Проверяем, что chapters содержит хотя бы один элемент
    if chapters:
        for i, chapter in enumerate(chapters):
            # Добавление главы, если она ещё не существует
            if i >= len(sections):
                sections.append({"title": "", "para": []})
            if i >= 2:  # Ограничиваем цикл двумя проходами для отладки
                break
            if len(chapter[0]) > 2:
                sections[i]['title'] = ta.translate(source_lang, target_lang, chapter[0], country)
            else:
                sections[i]['title'] = chapter[0]
            current_text = ''
            # print(f"Глава {i + 1}: {chapter[0]}")  # Выводим заголовок главы
            # print("Параграфы:")
            for j, paragraph in enumerate(chapter[1]):
                if len(paragraph) > max_len_chunk:
                    print("Paragraph size over limit")
                if len(current_text) < max_len_chunk and len(paragraph) < max_len_paragraph:
                    current_text = "".join([current_text, paragraph])
                else:
                    translated = ta.translate(source_lang, target_lang, current_text, country)
                    sections[i]['para'].append(translated)
                    current_text = paragraph  # Не забываем добавить текущий параграф в current_text после перевода

            # Если остался непереведённый текст в current_text
            if current_text:
                translated = ta.translate(source_lang, target_lang, current_text, country)
                sections[i]['para'].append(translated)

    # Сборщик книги в основной цикл
    new_book.chapters = sections
    new_book.write(f"./books/out_tst.fb2")



    # 4 Write FB2 new lang

    # Удалите exit(0), если вы хотите, чтобы программа продолжала выполнение
    exit(0)
