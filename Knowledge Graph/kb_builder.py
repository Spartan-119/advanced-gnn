from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_community.graphs.index_creator import GraphIndexCreator

# Load the .env file
load_dotenv()

# Initialize the ChatOpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))

# Question for the LLM
question = "When did Apple announce the Vision Pro?"
response = llm.predict(question)
print(f"Response: {response}")

# Create a graph from the text
text = "Apple announced the Vision Pro in 2023."
index_creator = GraphIndexCreator(llm=llm)
graph = index_creator.from_text(text)

# Display graph triples
print(graph.get_triples())
