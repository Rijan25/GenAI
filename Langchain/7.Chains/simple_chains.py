from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

prompt=PromptTemplate(
    template="Generate 5 interesting facts about the {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

chain=prompt | model | parser
result=chain.invoke({'topic':'Best Carrer in Australia.'})
print(result)

