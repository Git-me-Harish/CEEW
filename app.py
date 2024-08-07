# FINAL CHATBOT CODE

import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize chatbot components
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """ Answer the questions based on the provided context only. Please provide the most accurate response based on the question <context> {context} <context> Questions:{input} """
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = PyPDFLoader("breed2.pdf")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
final_documents = text_splitter.split_documents(docs)
# Extract text content from the Document instances
doc_texts = [doc.page_content for doc in final_documents]
embeddings_result = embeddings.embed_documents(doc_texts)
if embeddings_result:
    vectors = FAISS.from_documents(final_documents, embeddings)
else:
    raise ValueError("Failed to generate embeddings. Please check your input documents or try a different embedding model.")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    message = history[-1][0]
    start_time = time.time()
    response = retrieval_chain.invoke({'input': message})['answer']
    response_time = time.time() - start_time
    if response_time > 6:
        return [(f"Sorry, I couldn't generate a response within 6 seconds. Please try again.", None)]
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    chatbot.like(print_like_dislike, None, None)

    demo.queue()

if __name__ == "__main__":
    demo.launch()