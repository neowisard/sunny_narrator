

book.sections = list(
    map(
        lambda section: (section.header.title,section.epigraph,section.paragraphs,section.subtitle),
        book.sections,
    )
)

def fill_fb2_sections(book):
    book.sections = []
    book.sections = []

    for each in self.root.body.section.find_all(recursive=False):
        if not each.name in ['cite']:
            self._book._body['section']['text'].append(each.get_text(strip=True))
        else:
            match each.name:
                case "cite":
                    for eachOne in self.root.body.section.cite.find_all(recursive=False):
                        self._book._body['section']['cite'].append(eachOne.get_text(strip=True))
    for sections in book.sections:
        title = chapter.header.title
        epigraph
        paragraphs = section.paragraphs
        subtitle
        fb2.sectionss.append((title, paragraphs))