# THIS IS THE  FINAL CODE FOR CHATBOT

from flask import Flask, render_template, request, jsonify
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

app = Flask(__name__)

# Initialize chatbot components
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """ Answer the questions based on the provided context only. Please provide the most accurate response based on the question <context> {context} <context> Questions:{input} """
)
embeddings = OllamaEmbeddings(model='nomic-embed-text')
loader = PyPDFLoader("breed2.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])
embeddings_result = embeddings.embed_documents(final_documents)

if embeddings_result:
    vectors = FAISS.from_documents(final_documents, embeddings)
else:
    raise ValueError("Failed to generate embeddings. Please check your input documents or try a different embedding model.")

# Route to render index1.html for the chatbot
@app.route('/chatbot')
def chatbot_index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    # Your video feed logic here
    pass  # Placeholder to avoid indentation error

@app.route('/chat', methods=['POST'])
def chat():
    prompt1 = request.form['prompt']
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time :", time.process_time() - start)
        return jsonify({"response": response['answer']})

if __name__ == '__main__':
    app.run(debug=True)
    