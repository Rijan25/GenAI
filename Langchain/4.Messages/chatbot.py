from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.1,api_key=api_key)

messages=[
    SystemMessage(content="You are a helpful assistant acting as a chatbot."),
]

while True:
    user_input=input('')
    if user_input.lower()=='exit':
        break
    messages.append(HumanMessage(content=user_input))
    print(f'User:{user_input}')
    result=model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print(f'AI:{result.content}')

print(messages)
