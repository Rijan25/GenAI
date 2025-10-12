from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Generate me a twitter post on the learnings of {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate me a linkedin post on the learnings of {topic}',
    input_variables=['topic']
)

chains=RunnableParallel({
    'twitter_post':RunnableSequence(prompt1,model,parser),
    'linkedin_post':RunnableSequence(prompt2,model,parser)
})

result=chains.invoke({'topic':'MCP Servers'})
print(result['twitter_post'])
print('***********************************')
print(result['linkedin_post'])