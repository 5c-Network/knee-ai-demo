from typing import List
from pydantic import BaseModel


class Urls(BaseModel):
     urls : List[str]