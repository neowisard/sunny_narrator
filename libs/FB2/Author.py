from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Author:
    firstName: Optional[str] = None
    middleName: Optional[str] = None
    lastName: Optional[str] = None
    nickname: Optional[str] = None
    homePages: Optional[List[str]] = None
    emails: Optional[List[str]] = None
    id: Optional[str] = None
    """Holds information about author

    Attributes:
        firstName: author's first name
        middleName: author's middle name
        lastName: author's last name
        nickname: author's nickname
        homePages: list of author's websites
        emails: list of author's emails
        id: author's unique id
    """
