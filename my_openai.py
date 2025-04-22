from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # .env 자동 로드

class MyOpenAI(OpenAI):
    def __init__(self, **kwargs):
        super().__init__(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo-instruct",
            **kwargs
        )