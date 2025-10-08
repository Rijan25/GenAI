from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

class Person(BaseModel):
    name:str=Field(description='Name of the person.')
    age:int=Field(gt=0,lt=150,description='Age of the person.')
    city:str=Field(description='City of the Country.')

parser=PydanticOutputParser(pydantic_object=Person)   

template=PromptTemplate(
    template='Generate a random person name, age and city of the fictional {country} \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

parser2=StrOutputParser()

chain=template | model | parser

result=chain.invoke({'country':'Australia'})
print(result)