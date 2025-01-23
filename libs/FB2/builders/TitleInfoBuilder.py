import xml.etree.ElementTree as ET
from typing import Sequence, Union, Tuple, Optional
from datetime import datetime
from .BuildAuthorName import BuildAuthorName
from ..constants import FB2_LINK_PREFIX, GetLanguages
from ..Author import Author
from ..TitleInfo import TitleInfo


class TitleInfoBuilder:
    result: ET.Element

    def __init__(self,
                 rootTag: str = "title-info",
                 titleInfo: TitleInfo = None):
        """ Calls reset() method and, if titleInfo is set, adds all fields of
        titleInfo, except coverPageImages.

        Args:
            rootTag: tag name which will be used as name of root tag.
            titleInfo: class TitleInfo object, which will be converted to
                       FB2 title-info tag.
        """

        self.reset(rootTag)
        if titleInfo:
            self.AddGenres(titleInfo.genres)
            self.AddAuthors(titleInfo.authors)
            self.AddBookTitle(titleInfo.title)
            self.AddAnnotation(titleInfo.annotation)
            self.AddKeywords(titleInfo.keywords)
            self.AddDate(titleInfo.date)
            self.AddLang(titleInfo.lang)
            self.AddSrcLang(titleInfo.srcLang)
            self.AddTranslators(titleInfo.translators)
            self.AddSequences(titleInfo.sequences)

    def AddBookTitle(self, title: str):
        ET.SubElement(self.result, "book-title").text = title

    def AddGenres(self, genres: Sequence[str]) -> None:
        for genre in genres:
            ET.SubElement(self.result, "genre").text = genre

    def AddAuthors(self,
                   authors: Sequence[Union[Author, str, ET.Element]]) -> None:
        for author in authors:
            if isinstance(author, ET.Element):
                self.result.append(author)
            else:
                self.result.append(BuildAuthorName("author", author))

    def AddAnnotation(
        self,
        annotation: Optional[Union[str, Sequence[ET.Element]]]
    ) -> None:
        if annotation:
            annotationElement = ET.Element("annotation")
            if isinstance(annotation, str):
                for paragraph in annotation.split('\n'):
                    ET.SubElement(annotationElement, "p").text = paragraph
            else:
                for element in annotation:
                    annotationElement.append(element)
            self.result.append(annotationElement)

    def AddKeywords(self, keywords: Optional[Sequence[str]]) -> None:
        if keywords:
            ET.SubElement(self.result, "keywords").text = ", ".join(keywords)

    def AddDate(self, date: Optional[Tuple[datetime, Optional[str]]]) -> None:
        if date:
            ET.SubElement(
                self.result, "date", {"value": date[0].strftime("%Y-%m-%d")}
            ).text = date[1] or date[0].strftime("%d.%m.%Y")

    def AddLang(self, lang: str) -> None:
        if lang not in GetLanguages():
            raise ValueError(f"Unknown language {lang}")
        ET.SubElement(self.result, "lang").text = lang

    def AddSrcLang(self, srcLang: Optional[str]) -> None:
        if srcLang:
            if srcLang not in GetLanguages():
                raise ValueError(f"Unknown language {srcLang}")
            ET.SubElement(self.result, "src-lang").text = srcLang

    def AddTranslators(
        self,
        translators: Optional[Sequence[Union[str, ET.Element, Author]]]
    ) -> None:
        if translators:
            for translator in translators:
                if isinstance(translator, ET.Element):
                    self.result.append(translator)
                else:
                    self.result.append(
                        BuildAuthorName("translator", translator))

    def AddSequences(
        self,
        sequences: Optional[Sequence[Union[Tuple[str, int], ET.Element]]]
    ) -> None:
        if sequences:
            for sequence in sequences:
                if isinstance(sequence, ET.Element):
                    self.result.append(sequence)
                else:
                    ET.SubElement(self.result, "sequence", attrib={
                        "name": sequence[0],
                        "number": str(sequence[1])
                    })

    def AddCoverImages(self, coverLinks: Optional[Sequence[str]]) -> None:
        if coverLinks:
            coverPageElement = ET.Element("coverpage")
            for coverLink in coverLinks:
                ET.SubElement(coverPageElement, "image", attrib={
                    "{}:href".format(FB2_LINK_PREFIX): coverLink,
                    "alt": "Cover image"
                })
            self.result.append(coverPageElement)

    def GetResult(self) -> ET.Element:
        return self.result

    def reset(self, rootTag: str = "title-info") -> None:
        self.result = ET.Element(rootTag)
