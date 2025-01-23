from dataclasses import dataclass, field
from .Author import Author
from typing import List, Union, Optional, Tuple, Sequence
from datetime import datetime
from uuid import uuid4
import xml.etree.ElementTree as ET
from decimal import Decimal


@dataclass
class DocumentInfo:
    authors: List[Union[Author, str]] = field(default_factory=list)
    programUsed: Optional[str] = "Ae-Mc's FB2 library"
    date: Tuple[datetime, Optional[str]] = (datetime.now(), None)
    sourceUrl: Optional[str] = None
    sourceAuthor: Optional[Union[Author, str]] = None
    id: str = str(uuid4())
    version: Union[Decimal, str] = Decimal('1.0')
    history: Optional[Union[str, Sequence[ET.Element]]] = None
    publisher: Optional[Union[Author, str]] = None
    """Holds information about document such as version, used program, etc.

    Attributes:
        authors: document authors
        programUsed: program that book was created with
        date: document creation date
        sourceUrl: original document url (if exists)
        sourceAuthor:original document author (if exists)
        id: document unique identificator
        version: document version
        history: document changelog in free format
        publisher: publisher name if exists
    """
