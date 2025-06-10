from pydantic import BaseModel
from typing import List

class InferenceResponse(BaseModel):
    boxes: List[List[float]]
