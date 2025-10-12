from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('GOOGLE_API_KEY')
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite-preview-09-2025',temperature=0.1,api_key=api_key)

# Types of Messages
# 1.System Message : It is the initial prompt that we give to the LLM at the beginings.
# 2.Human Message : ThE queries that the human gives to the LLMs.
# 3.AI Message : The response or message the LLM replies.

messages=[
    SystemMessage(content='You are the helpful Assistant.'),
    HumanMessage(content='Who is the PM of India?')
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages[2].content)





