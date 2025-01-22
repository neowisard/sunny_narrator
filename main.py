from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from FB2 import FictionBook2, Author
from urllib import request
import dotenv
import yaml

# import libs.read.one as FBr
import libs.read.sec as fb2r
import libs.translate.llm as ta
from lxml.html.defs import table_tags

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
max_len_section = int(config_env['MAX_LEN_SECTION'])
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
    # result = FBr.FB2am(fb2file).parcer()
    orig_book = fb2r.fb2book(fb2file)
    new_book = FictionBook2()
    # Открыть файл и Распарсить файл

    # print(orig_book.get_title())
    # print(orig_book.get_description())
    # print(book.get_authors())
    # print(book.get_tags())
    # print(book.get_series())

    #command = f"Just translate next text form {source_lang} to {target_lang} language from {country} and text only translated string: "

    #new_book.titleInfo.title = ta.get_completion(orig_book.get_title(),'You are a helpful translator, linguist from {source_lang } to {target_lang} language')
    #new_book.titleInfo.title = ta.translate(source_lang, target_lang, orig_book.get_title(), country)
    #№new_book.titleInfo.annotation = ta.translate(source_lang, target_lang, orig_book.get_description(), country)
    '''
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
    print_book(new_book) 
    '''
    #print("New:\n")
    #print(new_book.titleInfo.title)4
    #print(new_book.titleInfo.annotation)
    #print(book.get_isbn())
    #print(book.get_title())
    #print(book.get_description())
    #print(book.get_lang())
    ##print(book.get_identifier())
    #print(book.get_series())
    #print(book.get_authors())
    #print(book.get_tags())
    #print(book.get_translators())

    # print(book.get_cover_image())
    #print(book.get_body_parsed())
    # Взать FBAM парсилку для body
    #body_with_tags = book.get_body_parsed()
    #body_with_tags_tuples = list(map(lambda x: (x[0], x[1]),body_with_tags))
    #print(book.get_body())
    # author_name = result._title_info['author'].get('last-name', '') + ' ' + result._title_info['author'].get('first-name', '')
    # book_title = result._title_info['book-title']
    # print(book_title)
    # 2 Анализаторы
    # Make vocab by spaCy
    # Add user vocab
    # Make classification

    # 3 Переводчики (с передачей system\add promt
    # Translate meta
    # Improve pictures
    # Chunking and translate body (stream)
    body, total_text_length = orig_book.get_body()
    count = len(body)
    #print_body_structure(body)
    print("Total text length in sections:", total_text_length, "Body 1 level tag (as section, title etc) ", count)


    def extract_chapters(body):
        chapters = []
        for element in body:
            tag_name = element[0]
            content = element[1]
            if tag_name == 'section':
                section_name = None
                section_subtitle = None
                paragraphs = []

                for sub_element in content:
                    sub_tag_name = sub_element[0]
                    if sub_tag_name == 'title':  #epigraph,title, subtitle,p
                        #Для title мы выдираем название (text) или если есть вложенные элементы то выдираем название из вложенных элементов
                        section_name = sub_element[1]
                        # Проверяем наличие текста для вложенного p для названия title
                        for sub_sub_element in sub_element[1]:
                            if sub_sub_element[0] == 'text':
                                section_name = sub_sub_element[1]
                            #elif sub_sub_element[0] == 'text':
                            #    #paragraphs.append(sub_sub_element[1])
                            #    section_name = sub_sub_element[1]
                            else:
                                sub_sub_content = sub_sub_element[1]
                                for sub_sub_sub_element in sub_sub_content:
                                    sub_sub_sub_tag_name = sub_sub_sub_element[0]
                                    if sub_sub_sub_tag_name == 'text':
                                        #paragraphs.append(sub_sub_sub_element[1])
                                        section_name = sub_sub_sub_element[1]
                                    elif sub_sub_sub_tag_name == 'subtitle':
                                        section_subtitle = sub_sub_sub_element[1]
                    elif sub_tag_name == 'text':
                        paragraphs.append(sub_element[1])
                    else:
                        sub_content = sub_element[1]
                        for sub_sub_element in sub_content:
                            sub_sub_tag_name = sub_sub_element[0]
                            if sub_sub_tag_name == 'text':
                                paragraphs.append(sub_sub_element[1])
                            elif sub_sub_tag_name == 'subtitle':
                                section_subtitle = sub_sub_element[1]
                            else:
                                sub_sub_content = sub_sub_element[1]
                                for sub_sub_sub_element in sub_sub_content:
                                    sub_sub_sub_tag_name = sub_sub_sub_element[0]
                                    if sub_sub_sub_tag_name == 'text':
                                        paragraphs.append(sub_sub_sub_element[1])
                                    #elif sub_sub_sub_tag_name == 'subtitle':
                                    #    section_subtitle = sub_sub_sub_element[1]

                if section_name is not None:
                    # Если есть subtitle, добавляем его к заголовку
                    if section_subtitle is not None:
                        section_name = f"{section_name}: {section_subtitle}"
                    chapters.append((section_name, paragraphs))
        return chapters


    # Пример использования
    body, total_text_length = orig_book.get_body()
    chapters = extract_chapters(body)

    # Убедитесь, что chapters содержит хотя бы 7 элементов
    if len(chapters) >= 2:
        for chapter in chapters[0:1]:  # Используем срез для получения 6 и 7 глав
            print(f"Title: {chapter[0]}")
            print("Paragraphs:")
            for paragraph in chapter[1]:
                print(f"  {paragraph}")
            print()
    else:
        print("В списке недостаточно глав для вывода.")
    # Translating over 2-3 LLM and translator.

    current_text = ''
    max_chunk = 0

    # Сборщик книги в основной цикл
    """ as example
 

    fb2.chapters = list(
        map(
            lambda chapter: (chapter.header.title, chapter.paragraphs),
            book.chapters,
        )
    )
    fb2.write(f"./Output/{fb2.titleInfo.title}.fb2") """





    # 4 Write FB2 new lang

    # Удалите exit(0), если вы хотите, чтобы программа продолжала выполнение
    exit(0)
