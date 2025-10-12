from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Generate a simple notes on the {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a simple Quiz Q&A from the following text {topic}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the given notes and the quizes into a single documents {notes} & {quizes}',
    input_variables=['notes','quizes']

)

parallel_chain=RunnableParallel({

    'notes':prompt1 | model | parser,
    'quizes': prompt2 | model | parser

})

merge_chain=prompt3 | model | parser

chain=parallel_chain | merge_chain
result=chain.invoke({'topic':'Quant Finance'})
print(result)