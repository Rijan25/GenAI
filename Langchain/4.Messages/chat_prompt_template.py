from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.1,api_key=os.getenv('GOOGLE_API_KEY'))

# chat template
chat_template=ChatPromptTemplate([
    ('system','You are a helpful {domain} assistant'),
    ('human','Give me the detail explaination of the {topic}')
])


prompt=chat_template.invoke({
    'domain':'AI',
    'topic':'MCP Servers'
})

response=model.invoke(prompt)
print(response)