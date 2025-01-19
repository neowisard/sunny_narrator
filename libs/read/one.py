from bs4 import BeautifulSoup as Soup


class FB2am:
    class _book:
        _title_info = {
            'sequence': {},
            'author': {}
        }
        _body = {
            'title': [],
            'section': {
                'cite': [],
                'text': []
            }
        }

    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as xml:
            self.root = Soup(xml.read(), 'xml')

    def parcer(self):
        for each in self.root.description.find('title-info').find_all(recursive=False):
            if not each.name in ['author', 'sequence']:
                self._book._title_info[each.name] = each.get_text()
            else:
                match each.name:
                    case "author":
                        for eachOne in self.root.description.author.find_all(recursive=False):
                            self._book._title_info['author'][eachOne.name] = eachOne.get_text(strip=True)
                    case "sequence":
                        self._book._title_info['sequence']['number'] = each['number']
                        self._book._title_info['sequence']['name'] = each['name']
        for each in self.root.body.title.find_all(recursive=False):
            self._book._body['title'].append(each.get_text(strip=True))
        for each in self.root.body.section.find_all(recursive=False):
            if not each.name in ['cite']:
                self._book._body['section']['text'].append(each.get_text(strip=True))
            else:
                match each.name:
                    case "cite":
                        for eachOne in self.root.body.section.cite.find_all(recursive=False):
                            self._book._body['section']['cite'].append(eachOne.get_text(strip=True))

        return self._book
