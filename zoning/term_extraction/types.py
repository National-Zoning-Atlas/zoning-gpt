from pydantic import BaseModel

class District(BaseModel):
    full_name: str
    short_name: str
