from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.2,api_key=api_key)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Generate me a joke on the topic of {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain me this joke {text}',
    input_variables=['text']
)

sequence_runnable=RunnableSequence(prompt1,model,parser)

parallel_runnable=RunnableParallel({
    'passthrough':RunnablePassthrough(),
    'joke_explanation': RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(sequence_runnable,parallel_runnable)
result=final_chain.invoke({'topic':'AI'})
print(result['passthrough'])
print(result['joke_explanation'])
