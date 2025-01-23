import xml.etree.ElementTree as ET
from typing import Union
from ..Author import Author  # type: ignore


def BuildAuthorName(rootTag: str, author: Union[Author, str]) -> ET.Element:
    rootElement = ET.Element(rootTag)
    if isinstance(author, str):
        authorNameParts = author.split(' ')
        if len(authorNameParts) == 1:
            ET.SubElement(rootElement, "nickname").text = authorNameParts[0]
        elif len(authorNameParts) == 2:
            ET.SubElement(rootElement, "first-name").text = authorNameParts[0]
            ET.SubElement(rootElement, "last-name").text = authorNameParts[1]
        elif len(authorNameParts) == 3:
            ET.SubElement(rootElement, "first-name").text = authorNameParts[0]
            ET.SubElement(rootElement, "last-name").text = authorNameParts[1]
            ET.SubElement(rootElement, "middle-name").text = authorNameParts[2]
    else:
        if author.nickname is not None:
            ET.SubElement(rootElement, "nickname").text = author.nickname
        if author.firstName is not None:
            ET.SubElement(rootElement, "first-name").text = author.firstName
        if author.middleName is not None:
            ET.SubElement(rootElement, "middle-name").text = author.middleName
        if author.lastName is not None:
            ET.SubElement(rootElement, "last-name").text = author.lastName
        if author.id is not None:
            ET.SubElement(rootElement, "id").text = author.id
        if author.homePages:
            for homePage in author.homePages:
                ET.SubElement(rootElement, "home-page").text = homePage
        if author.emails:
            for email in author.emails:
                ET.SubElement(rootElement, "email").text = email
    return rootElement
