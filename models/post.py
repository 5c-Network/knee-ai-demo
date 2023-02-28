from typing import List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field


class Post(BaseModel):
    """Post model has only two fields (resources, id) which will be
    used by the animal-detection model
    """
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    resources: Optional[List]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
