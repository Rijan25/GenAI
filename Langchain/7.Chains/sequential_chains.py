from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Give me a detail report on the {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the following text in 3 bullet points {text}',
    input_variables=['text']
)

chain=prompt1 | model | parser | prompt2 | model | parser
result=chain.invoke({'topic':'Democarcy is Scam.'})
print(result)