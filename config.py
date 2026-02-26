from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel

load_dotenv()
# USE IN CASE YOU HAVE OPENAI KEY

class Config(BaseModel):
    API_KEY: str

config = Config(
    API_KEY=getenv("OPENAI_API_KEY")
)