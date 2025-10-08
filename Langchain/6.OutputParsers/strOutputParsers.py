from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

parser=StrOutputParser()

template1=PromptTemplate(
    template='Write a detail report on {topic}.',
    input_varibales=['topic']
)

template2=PromptTemplate(
    template='Write a 5 line summary on the following text {text}.',
    input_variables=['text']
)

chain= template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'NEPSE'})
print(result)







