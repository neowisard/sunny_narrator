from dataclasses import dataclass

from .FB2Builder import FB2Builder
from .FictionBook2dataclass import FictionBook2dataclass


@dataclass
class FictionBook2(FictionBook2dataclass):
    """Represents fb2 book

    Attributes:
        stylesheets: optional css styles
        titleInfo: book title information
        sourceTitleInfo: source book title information (if exists)
        documentInfo: document information
        customInfos: free format additional information
        chapters: list of chapters names and lists of paragraphs or Elements
            with paragraphs or lists of Elements

    Methods:
        write: writes to fb2 file
    """

    def write(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(self))

    def __str__(self) -> str:
        return FB2Builder._PrettifyXml(FB2Builder(self).GetFB2())
