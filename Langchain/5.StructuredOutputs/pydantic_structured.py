from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal,Optional
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

class Review(BaseModel):
    key_themes:list[str]=Field(description='Key theme of the product or the movie.')
    summary:str=Field(description='Summary of the review.')
    sentiment:Literal["pos","neg"]=Field(description='Sentiment of the movie or the product.')
    rating:float=Field(default=0.0,gt=0,lt=5,description='Rating of the movie between 0-5')
    name:Optional[str]=Field(default=None,description='Name of the reviewer or complain giver.')

structured_model=model.with_structured_output(Review)

response=structured_model.invoke('Bahubali is the best movie of this decade. The performance of the prabhas and other side actors are superbs . It is going to break all the records of the indian cinema. I will give it 4.5 out of 5 . This review is given by Rijan Pokhrel.')
print(response)
print(response.summary)
