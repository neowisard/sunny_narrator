import xml.etree.ElementTree as ET
import libs.translate.llm as ta
from icecream import ic
import os
import re
from dotenv import load_dotenv
from pathlib import Path

config_dir = Path(__file__).parent.resolve() / "config"

# load .env config
load_dotenv(config_dir / ".env")

api_key = os.getenv('API_KEY', '')  # Значение по умолчанию - пустая строка
base_url = os.getenv('API_BASE', 'https://api.openai.com/v1')
spromt = os.getenv('S_PROMT', 'You are a helpful assistant.')
model = os.getenv('MODEL', 'gpt-4-turbo')
api_key2 = os.getenv('API_KEY2', '')  # Значение по умолчанию - пустая строка
base_url2 = os.getenv('API_BASE2', 'https://api.openai.com/v1')
spromt2 = os.getenv('S_PROMT2', 'You are a helpful assistant.')
model2 = os.getenv('MODEL2', 'gpt-4-turbo')
api_timeout = int(os.getenv('TIMEOUT', 6000))
api_timeout2 = int(os.getenv('TIMEOUT2', 6000))
vocab = os.getenv('VOCAB', 'config/dict.env')
source_lang = os.getenv('SOURCE_LANG')
target_lang = os.getenv('TARGET_LANG')
country = os.getenv('COUNTRY')
max_len_chunk = int(os.getenv('MAX_LEN_CHUNK'))
myfile = Path(__file__).parent.resolve() / os.getenv('FILE', 'books/ExampleBook.fb2')
debug = os.getenv('DEBUG', 'False').lower() in ['true', '1', 't']


if spromt.lower() == 'none':
    spromt = None
if spromt2.lower() == 'none':
    spromt2 = None

#sunny-narrator code by neowisard

def load_vocab_from_file(file_path, source_lang, target_lang):
    vocab = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and '=' in line:
                source_word, target_word = line.split('=', 1)
                vocab[source_word] = target_word
    return vocab









def parse_xml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        start_body = content.find('<body')
        end_body = content.find('</body>') + len('</body>')

        header = content[:start_body]
        body = content[start_body:end_body]
        footer = content[end_body:]

        if start_body == -1 or end_body == -1:
            raise ValueError("Body tag not found in the XML file")

        # Удаление пространств имен
        content = re.sub(r'\sxmlns="[^"]+"', '', content, count=1)

        parser = ET.XMLParser()
        root = ET.fromstring(content, parser=parser)

        # Парсим тело без учета пространств имен
        body_element = root.find('.//body')

        new_body_element = ET.Element('body')
        for child in body_element:
            new_body_element.append(child)
        if body_element is None:
            raise ValueError("Body element not found in the XML file.")

        return new_body_element, header, footer

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def chunkify_fb2(body, max_len_chunk, source_lang, target_lang, country):
    body_str = ET.tostring(body, encoding='unicode')
    chunks = []
    start = 0
    end_tags = {'</p>', '</section>', '</subtitle>', '<empty-line/>', '</P>', '</SECTION>', '</SUBTITLE>', '</ns0:p>', '</ns0:section>', '</ns0:subtitle>'}
    style='xml'
    outline_text = ""
    while start < len(body_str):
        end = start + max_len_chunk
        if end >= len(body_str):
            end = len(body_str)
        else:
            # Ищем один из end_tags в строке, начиная с позиции end и идя назад
            found_tag = None
            for tag in end_tags:
                pos = body_str.rfind(tag, start, end)
                if pos != -1 and (found_tag is None or pos > found_tag[1]):
                    found_tag = (tag, pos)

            if found_tag:
                end = found_tag[1] + len(found_tag[0])

        source_text = body_str[start:end]
        try:
            translated_chunk, outline = ta.translate(source_lang, target_lang, source_text,  style, outline_text, country)
            if translated_chunk is not None:
                #translated_tree = ET.fromstring(translated_chunk)
                # Удаляем пространства имен
                #remove_namespaces(translated_tree)
                # Преобразуем обратно в строку без пространств имен
                chunks.append(translated_chunk)
                outline_text = outline
            else:
                raise ValueError(f"Translation failed for chunk: {source_text}")
        except Exception as e:
            raise ValueError(f"Error during translation: {e}")
        start = end

    return chunks

def chunkify_text(text, max_len_chunk, source_lang, target_lang, country):
    chunks = []
    outline_text = ""
    start = 0
    end = 0
    text_len = len(text)
    style = 'txt'
    while start < text_len:
        end = start + max_len_chunk
        if end >= text_len:
            end = text_len
        else:
            # Ищем точку или переход строки в строке, начиная с позиции end и идя назад
            while end > start and text[end] not in {'.', '\n'}:
                end -= 1

            # Если не нашли точку или переход строки, то разбиваем по max_len_chunk
            if end == start:
                end = start + max_len_chunk

        source_text = text[start:end]
        try:
            translated_chunk, outline  = ta.translate(source_lang, target_lang, source_text, style, outline_text, country)
            outline_text = outline
            if translated_chunk is not None:
                chunks.append(translated_chunk)
            else:
                raise ValueError(f"Translation failed for chunk: {source_text}")
        except Exception as e:
            raise ValueError(f"Error during translation: {e}")
        start = end

    return chunks

def write_chunks_to_file(header, chunks, footer, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        for chunk in chunks:
            f.write(chunk + '\n')
        f.write(footer)


#/// Написать функцию перевода чанков здесь, перенеся ее из chunkify и вынес из app.translate и добавив словари  логику чтобы сделать синхронную\асинхронную на 2 LLM
#/// Это также поможет создать шкалу обработки
#/// в конец добавить notification



def main():

    file_name, file_extension = os.path.splitext(os.path.basename(myfile))
    file_name_without_ext = file_name
    output_dir = os.path.dirname(myfile)
    vocab_dict = load_vocab_from_file(vocab, source_lang, target_lang)
    if debug:
        ic(vocab_dict)

    if file_extension.lower() == '.fb2':
        output_file = f"{output_dir}/{file_name_without_ext}_{target_lang}.fb2"
        # Вызов функции для обработки fb2 файла
        body, header, footer = parse_xml(myfile)
        # ta_header = ta.translate(source_lang, target_lang, header, country)
        chunks = chunkify_fb2(body, max_len_chunk, source_lang, target_lang, country)
        write_chunks_to_file(header, chunks, footer, output_file)

    elif file_extension.lower() == '.txt':
        output_file = f"{output_dir}/{file_name_without_ext}_{target_lang}.txt"
        text = read_txt_file(myfile)
        # Вызов функции для обработки txt файла
        chunks = chunkify_text(text, max_len_chunk, source_lang, target_lang, country)
        header=''
        footer=''
        write_chunks_to_file(header, chunks, footer, output_file)
    else                                                           :
        raise ValueError(f"Unsupported file extension: {file_extension}")


if __name__ == '__main__':
    main()