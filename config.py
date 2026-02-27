from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel

load_dotenv()

class Config(BaseModel):
    TOKEN: str # Bot token for telegram

config = Config(
    TOKEN=getenv("TOKEN")
)