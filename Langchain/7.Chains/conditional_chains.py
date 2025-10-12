from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

parser=StrOutputParser()

class Review(BaseModel):
    sentiment:Literal['Positive','Negative']=Field(description="The sentiment of the review.")

review_parser=PydanticOutputParser(pydantic_object=Review)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Give the sentiment of the following review {review} in the {format_instructions}',
    input_varibales=['review'],
    partial_variables={'format_instructions':review_parser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template='Write an appropriate 2 line  response to this positive feedback {feedback}.',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write an appropriate 2 line response to this negative feedback {feedback}.',
    input_variables=['feedback']
)

classifer_chain=prompt1 | model | review_parser

branch_chain=RunnableBranch(
    (lambda x: x.sentiment=='Positive', prompt2 | model | parser),
    (lambda x: x.sentiment=='Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "No sentiment detected")
)

chain=classifer_chain | branch_chain
result=chain.invoke({'review':'The product quality is really shit I want my money back.'})
print(result)


