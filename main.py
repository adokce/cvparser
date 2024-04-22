import os
import tempfile
import logging
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI, ChatPerplexity
from langchain_community.llms.replicate import Replicate
from openai import Client
from langchain.llms.base import LLM

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the OpenAI API key
# os.environ['OPENAI_API_KEY'] is already set via replit secrets
# also:
# os.environ['PPLX_API_KEY'] is also already set

openai_api_key = os.environ['OPENAI_API_KEY']
pplx_api_key = os.environ['PPLX_API_KEY']
replicate_api_token = os.environ['REPLICATE_API_TOKEN']

client = Client(api_key=openai_api_key)

# Read the default magic prompt from the file
with open("prompt.txt", "r") as file:
  default_magic_prompt = file.read().strip()

# Set the maximum file size (in bytes)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html',
                         default_magic_prompt=default_magic_prompt)


@app.route('/upload', methods=['POST'])
def upload_and_process_pdf():
  logging.info('Received request to /upload endpoint')

  if 'pdf' not in request.files:
    logging.error('No PDF file found in the request')
    return jsonify({"error": "No PDF file found in the request"}), 400
  file = request.files['pdf']
  if file.filename == '':
    logging.error('No file selected')
    return jsonify({"error": "No file selected"}), 400

  selected_model = request.form.get('model', 'gpt-3.5-turbo')
  logging.info(f'Selected model: {selected_model}')

  magic_prompt = request.form.get('magic_prompt', default_magic_prompt)

  if file and file.filename.endswith('.pdf'):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_file.write(file.read())
      temp_file_path = temp_file.name

    try:
      logging.info('Loading and splitting PDF content')
      # Load and split the PDF content
      loader = PyPDFLoader(temp_file_path)
      documents = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                     chunk_overlap=0)
      texts = text_splitter.split_documents(documents)

      logging.info('Creating embeddings and vector store')
      # Create a new instance of OpenAIEmbeddings for each request
      embeddings = OpenAIEmbeddings()

      # Create a new vector store for each request
      vector_store = Chroma.from_documents(texts, embeddings)

      logging.info('Setting up question-answering chain')
      # Set up the question-answering chain with the selected model
      if selected_model == 'llama-3-8b-instruct':
        qa = RetrievalQA.from_chain_type(llm=ChatPerplexity(
            client=client, model=selected_model, pplx_api_key=pplx_api_key),
                                         chain_type="stuff",
                                         retriever=vector_store.as_retriever())
        # there is some bug in this here you get bad results. it's disabled in frontend select
      elif selected_model == 'meta/meta-llama-3-8b-instruct':
        qa = RetrievalQA.from_chain_type(
            llm=Replicate(model="meta/meta-llama-3-8b-instruct"),
            chain_type="stuff",
            retriever=vector_store.as_retriever())
      else:
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model=selected_model),
                                         chain_type="stuff",
                                         retriever=vector_store.as_retriever())

      logging.info('Asking question and getting answer')
      # Ask the question and get the answer
      answer = qa.invoke(magic_prompt)

      logging.info('Returning answer')
      return jsonify({"summary": answer}), 200
    except Exception as e:
      logging.exception('An error occurred during processing')
      return jsonify({"error": str(e)}), 500
    finally:
      # Delete the temporary file
      os.unlink(temp_file_path)
  else:
    logging.error('Invalid file type. Only PDF files are allowed.')
    return jsonify({"error":
                    "Invalid file type. Only PDF files are allowed."}), 400


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
