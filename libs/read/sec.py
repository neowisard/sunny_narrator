import os
from calendar import error
from ftplib import error_reply
import re

from bs4 import BeautifulSoup
#import lxml as result

__all__ = ['get_fb2'],


def _get_file(file):
    file = None
    if file.lower().endswith(('.fb2')):
        file = fb2book(file)
    return file


def get_fb2(file):
    _get_file(file)


class fb2book:

    def __init__(self, file):
        self.file = file
        with open(file, 'r+', encoding='utf-8') as fb2file:
            fb2_content = fb2file.read()
        self.soup = BeautifulSoup(fb2_content, "xml")
        self.body = self.soup.find('body') if self.soup.find('body') else None

    def get_body(self):
        if self.body:
            return self.extract_elements(self.body)
        return []

    def clean_text(self,text):
        # Удаление всех переносов строк
        text = re.sub(r'\n+', ' ', text)
        # Удаление всех последовательностей из трех и более пробелов
        text = re.sub(r' {3,}', ' ', text)
        # Удаление лишних пробелов в начале и конце строки
        text = text.strip()
        return text

    def extract_elements(self, element):
        result = []
        section_text_length = 0

        for child in element.children:
            if child.name:  # Если это тег
                if child.name == 'section':
                    section_result, section_length = self.extract_elements(child)
                    section_text_length += section_length
                    result.append((child.name, section_result, section_length))
                else:
                    child_result, child_length = self.extract_elements(child)
                    section_text_length += child_length
                    result.append((child.name, child_result))
            elif child.string and child.string.strip():  # Если это текст
                text = child.string.strip()
                section_text_length += len(text)
                result.append(('text', text))

        return result, section_text_length


    def extract_chapters(self,body):
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
                    sub_content = sub_element[1]

                    if sub_tag_name == 'title':
                        # Обработка title
                        if isinstance(sub_content, str):
                            section_name = self.clean_text(sub_content)
                        else:
                            for sub_sub_element in sub_content:
                                sub_sub_tag_name = sub_sub_element[0]
                                if sub_sub_tag_name == 'text':
                                    section_name = self.clean_text(sub_sub_element[1])
                                elif sub_sub_tag_name == 'subtitle':
                                    section_subtitle = self.clean_text(sub_sub_element[1])
                                else:
                                    sub_sub_content = sub_sub_element[1]
                                    for sub_sub_sub_element in sub_sub_content:
                                        sub_sub_sub_tag_name = sub_sub_sub_element[0]
                                        if sub_sub_sub_tag_name == 'text':
                                            section_name = self.clean_text(sub_sub_sub_element[1])
                                        elif sub_sub_sub_tag_name == 'subtitle':
                                            section_subtitle = self.clean_text(sub_sub_sub_element[1])

                    elif sub_tag_name == 'subtitle':
                        # Обработка subtitle
                        if isinstance(sub_content, str):
                            section_subtitle = self.clean_text(sub_content)
                        else:
                            for sub_sub_element in sub_content:
                                if sub_sub_element[0] == 'text':
                                    section_subtitle = self.clean_text(sub_sub_element[1])

                    elif sub_tag_name == 'p':
                        # Обработка параграфа
                        paragraph_text = []  # Временная переменная для накопления текста внутри параграфа
                        if isinstance(sub_content, str):
                            paragraph_text.append(self.clean_text(sub_content))
                        else:
                            for sub_sub_element in sub_content:
                                if sub_sub_element[0] == 'text':
                                    paragraph_text.append(self.clean_text(sub_sub_element[1]))
                                elif sub_sub_element[0] == 'emphasis':
                                    # Обработка вложенного emphasis их выделяем кавычками
                                    for sub_sub_sub_element in sub_sub_element[1]:
                                        if sub_sub_sub_element[0] == 'text':
                                            cleaned_text = self.clean_text(sub_sub_sub_element[1])
                                            paragraph_text.append(f'"{cleaned_text}"')
                                            # paragraph_text.append(self.clean_text(sub_sub_sub_element[1]))
                                else:
                                    sub_sub_content = sub_sub_element[1]
                                    for sub_sub_sub_element in sub_sub_content:
                                        if sub_sub_sub_element[0] == 'text':
                                            paragraph_text.append(self.clean_text(sub_sub_sub_element[1]))
                                            # Все равно небольшой косяк , пример (пробел после обработки тега strong )
                                            #   T HE BEAR WAS A BIG BOAR GRIZZLY DOWN OUT OF CANADA. Th

                        # Объединяем все части текста в одну строку и добавляем в список paragraphs
                        paragraphs.append(' '.join(paragraph_text))

                    # В код ниже надо тоже джойнить строку с параграфом
                    else:
                        # Обработка всех остальных тегов
                        for sub_sub_element in sub_content:
                            if sub_sub_element[0] == 'text':
                                paragraphs.append(self.clean_text(sub_sub_element[1]))
                            # elif sub_sub_element[0] == 'subtitle':
                            #    section_subtitle = self.clean_text(sub_sub_element[1])
                            else:
                                sub_sub_content = sub_sub_element[1]
                                for sub_sub_sub_element in sub_sub_content:
                                    if sub_sub_sub_element[0] == 'text':
                                        paragraphs.append(self.clean_text(sub_sub_sub_element[1]))

                if section_name is not None:
                    # Если есть subtitle, добавляем его к заголовку
                    if section_subtitle is not None:
                        section_name = f"{section_name}"
                    chapters.append((section_name, paragraphs))
        return chapters


    def get_identifier(self):
        return self.soup.find('id').text if self.soup.find('id') else None

    def get_title(self):
        return self.soup.find('book-title').text if self.soup.find('book-title') else None

    def get_authors(self):
        authors = []
        for author in self.soup.find_all('author'):
            first_name = author.find('first-name').text if author.find('first-name') else ''
            last_name = author.find('last-name').text if author.find('last-name') else ''
            middle_name = author.find('middle-name').text if author.find('middle-name') else ''
            authorFL = ' '.join(filter(None, [first_name, middle_name, last_name]))
            if authorFL:
                authors.append(authorFL)
        return authors

    def get_translators(self):
        translators = []
        for translator in self.soup.find_all('translator'):
            first_name = translator.find('first-name').text
            last_name = translator.find('last-name').text
            if first_name != None:
                translatorsFL = ' '.join(filter(None, [first_name, last_name]))
                translators.append({translatorsFL})
        return translators

    def get_series(self):
        return self.soup.find('sequence')['name'] if self.soup.find('sequence') else None

    def get_lang(self):
        return self.soup.find('lang').text if self.soup.find('lang') else None

    def get_description(self):
        return self.soup.find('annotation').text if self.soup.find('annotation') else None

    def get_tags(self):
        return [genre.text for genre in self.soup.find_all('genre')]

    def get_isbn(self):
        return self.soup.find('isbn').text if self.soup.find('isbn') else None

    def get_cover_image(self):
        return self.soup.find('binary', {'content-type': 'image/jpeg'}) or self.soup.find('binary',
                                                                                          {'content-type': 'image/png'})

    def save_cover_image(cover_image, cover_image_type, output_dir='output'):
        """Сохраняет файл обложки."""
        cover_image_name = cover_image['id']
        cover_image_data = cover_image.text
        cover_image_path = os.path.join(output_dir, f'{cover_image_name}.{cover_image_type}')

        os.makedirs(output_dir, exist_ok=True)
        with open(cover_image_path, 'wb') as img_file:
            img_file.write(bytearray.fromhex(cover_image_data))
        return cover_image_name, cover_image_type

    def save_body_as_html(self, output_dir='output', output_file_name='body'):
        """Сохраняет тело книги в HTML файл."""
        body_path = os.path.join(output_dir, output_file_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(body_path, 'w+', encoding='utf-8') as html_file:
            html_file.write(self.body)
        return body_path
