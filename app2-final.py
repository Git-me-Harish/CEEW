import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Load the YOLO model
model = YOLO("version4c.pt")

# Function to predict image
def predict_image(img, conf_threshold, iou_threshold):
    # Perform object detection
    results = model.predict(
        source=img, conf=conf_threshold, iou=iou_threshold, show_labels=True, show_conf=True, imgsz=640
    )
    # Plot the result
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
    return im

# Initialize chatbot components
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """ Answer the questions based on the provided context only. Please provide the most accurate response based on the question <context> {context} <context> Questions:{input} """
)
embeddings = OllamaEmbeddings(model='nomic-embed-text')
loader = PyPDFLoader("breed2.pdf")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
final_documents = text_splitter.split_documents(docs)
embeddings_result = embeddings.embed_documents(final_documents)
if embeddings_result:
    vectors = FAISS.from_documents(final_documents, embeddings)
else:
    raise ValueError("Failed to generate embeddings. Please check your input documents or try a different embedding model.")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Function to handle both image prediction and chatbot
def predict_and_chat(img, conf_threshold, iou_threshold, prompt):
    # Predict image
    prediction_result = predict_image(img, conf_threshold, iou_threshold)
    
    # Chat
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time :", time.process_time() - start)
    chat_response = response["answer"]
    
    return prediction_result, chat_response

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Image")
            conf_threshold = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
            iou_threshold = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
            query_input = gr.Textbox(lines=7, label="Enter your question")
            run_button = gr.Button("Run")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Result")
            output_text = gr.Textbox(label="Response")
    run_button.click(predict_and_chat, inputs=[img_input, conf_threshold, iou_threshold, query_input], outputs=[output_image, output_text])

demo.launch(debug=True)