from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Write a joke on the topic of {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain me this joke {text}',
    input_variables=['text']
)

chains=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result=chains.invoke({'topic':'AI'})
print(result)
