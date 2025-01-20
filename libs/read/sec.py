import os
from calendar import error
from ftplib import error_reply

from bs4 import BeautifulSoup

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
        self.body = self.soup.find('body').prettify() if self.soup.find('body') else None

    def get_identifier(self):
        return self.soup.find('id').text if self.soup.find('id') else None

    def get_title(self):
        return self.soup.find('book-title').text if self.soup.find('book-title') else None

    def get_body(self):
        return self.soup.find('body').text if self.soup.find('body') else exit(3)
        return self.body

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
