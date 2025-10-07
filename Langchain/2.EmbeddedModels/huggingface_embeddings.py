from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text='Who is Narendra Modi?'

vector=embedding.embed_query(text)
# print(vector)
print(len(vector))