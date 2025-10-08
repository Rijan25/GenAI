from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

template=PromptTemplate(
    template='Who is the PM of {country}?',
    input_variables=['country'],
    validate_template=True
)

chain=template | model 
response=chain.invoke({'country':'Nepal'})
print(response.content)

