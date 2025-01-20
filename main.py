from pathlib import Path

import dotenv
import yaml

# import libs.read.one as FBr
import libs.read.sec as fb2r
from hypothesis import example

# from fb2reader2 import FictionBook2, Author
# import fb2reader

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

if __name__ == '__main__':
    # 1 Получить переменные и файл
    # result = FBr.FB2am(fb2file).parcer()
    book = fb2r.fb2book(fb2file)
    # Открыть файл и Распарсить файл
    print(book.get_isbn())
    print(book.get_title())
    print(book.get_description())
    print(book.get_lang())
    print(book.get_identifier())
    print(book.get_series())
    print(book.get_authors())
    print(book.get_tags())
    print(book.get_translators())

    # print(book.get_cover_image())
    print(book.get_body())
    # Взать FBAM парсилку для body

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
    # Translating over 2-3 LLM and translator.

    current_text = ''
    max_chunk = 0
    #Previous version (worked)
    # print(len(result._body['section']['text']))
    # for i in range(3):
    #    max_chunk = max_len_chunk+len(result._body['section']['text'][i])
    #    if (len(current_text) < max_chunk) :
    #        current_text = current_text + '\n' + result._body['section']['text'][i]
    #        print(i, "\n")
    #    else:
    #        translation = ta.translate(source_lang, target_lang, current_text, country)
    #        print(translation)
    #        current_text = ''  # Сброс текущего текста после перевода

    # Сборщик книги в основной цикл
    """ as example
    print(book.header)
    print(
        "-----------------\nTable of Contents\n-----------------",
        end="\n",
    )
    for chapterHeader in book.header.tableOfContents:
        print(chapterHeader)
    fb2 = FictionBook2()
    fb2.titleInfo.title = book.header.title
    fb2.titleInfo.authors = cast(
        List[Union[str, Author]],
        book.header.authors,
    )
    fb2.titleInfo.annotation = book.header.annotation
    fb2.titleInfo.genres = book.header.genres
    fb2.titleInfo.lang = target_lang
    fb2.titleInfo.sequences = (
        [(book.header.sequence.name, book.header.sequence.number)]
        if book.header.sequence
        else None
    )
    fb2.titleInfo.keywords = book.header.tags
    fb2.titleInfo.coverPageImages = (
        [book.header.coverImageData]
        if book.header.coverImageData
        else None
    )
    fb2.titleInfo.date = (book.header.publicationDate, None)

    fb2.chapters = list(
        map(
            lambda chapter: (chapter.header.title, chapter.paragraphs),
            book.chapters,
        )
    )
    fb2.write(f"./Output/{fb2.titleInfo.title}.fb2")
    await Logoff(client)
print(f"All requests took {time() - t} seconds.")

asyncio.get_event_loop().run_until_complete(main()) """




    # 4 Write FB2 new lang

    # Удалите exit(0), если вы хотите, чтобы программа продолжала выполнение
    exit(0)
