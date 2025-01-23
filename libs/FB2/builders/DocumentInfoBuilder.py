import xml.etree.ElementTree as ET
from typing import Union, Sequence, Tuple, Optional
from datetime import datetime
from decimal import Decimal
from ..Author import Author
from ..DocumentInfo import DocumentInfo
from .BuildAuthorName import BuildAuthorName


class DocumentInfoBuilder:
    result: ET.Element

    def __init__(self, documentInfo: DocumentInfo = None):
        self.reset()
        if documentInfo:
            self.AddAuthors(documentInfo.authors)
            self.AddProgramUsed(documentInfo.programUsed)
            self.AddDate(documentInfo.date)
            self.AddSourceUrl(documentInfo.sourceUrl)
            self.AddSourceAuthor(documentInfo.sourceAuthor)
            self.AddId(documentInfo.id)
            self.AddVersion(documentInfo.version)
            self.AddHistory(documentInfo.history)
            self.AddPublisher(documentInfo.publisher)

    def AddAuthors(self,
                   authors: Sequence[Union[Author, str, ET.Element]]) -> None:
        for author in authors:
            if isinstance(author, ET.Element):
                self.result.append(author)
            else:
                self.result.append(BuildAuthorName("author", author))

    def AddProgramUsed(self, programUsed: Optional[str]) -> None:
        if programUsed:
            ET.SubElement(self.result, "program-used").text = programUsed

    def AddDate(self, date: Tuple[datetime, Optional[str]]) -> None:
        ET.SubElement(
            self.result, "date", {"value": date[0].strftime("%Y-%m-%d")}
        ).text = date[1] or date[0].strftime("%d.%m.%Y")

    def AddSourceUrl(self, sourceUrl: Optional[str]) -> None:
        if sourceUrl:
            ET.SubElement(self.result, "src-url").text = sourceUrl

    def AddSourceAuthor(self, sourceAuthor: Optional[Union[Author, str]]):
        if sourceAuthor:
            self.result.append(BuildAuthorName("src-ocr", sourceAuthor))

    def AddId(self, id: str) -> None:
        ET.SubElement(self.result, "id").text = id

    def AddVersion(self, version: Union[Decimal, str]) -> None:
        ET.SubElement(self.result, "version").text = str(Decimal(version))

    def AddHistory(
            self, history: Optional[Union[str, Sequence[ET.Element]]]) -> None:
        if history:
            if isinstance(history, str):
                ET.SubElement(self.result, "history").text = history
            else:
                historyElement = ET.SubElement(self.result, "history")
                for element in history:
                    historyElement.append(element)

    def AddPublisher(self, publisher: Optional[Union[Author, str]]) -> None:
        if publisher:
            self.result.append(BuildAuthorName("publisher", publisher))

    def GetResult(self) -> ET.Element:
        return self.result

    def reset(self) -> None:
        self.result = ET.Element("document-info")
