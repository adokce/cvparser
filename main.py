import os
import tempfile
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import contextvars

app = Flask(__name__)

# Set up the OpenAI API key
# os.environ['OPENAI_API_KEY'] is already set via replit secrets

# Set up the embedding model
embeddings = OpenAIEmbeddings()
vector_store_var = contextvars.ContextVar('vector_store')

# Read the magic prompt from the file
with open("prompt.txt", "r") as file:
  magic_prompt = file.read().strip()

# Set the maximum file size (in bytes)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_and_process_pdf():
  if 'pdf' not in request.files:
    return jsonify({"error": "No PDF file found in the request"}), 400
  file = request.files['pdf']
  if file.filename == '':
    return jsonify({"error": "No file selected"}), 400

  selected_model = request.form.get('model', 'gpt-3.5-turbo')

  if file and file.filename.endswith('.pdf'):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_file.write(file.read())
      temp_file_path = temp_file.name

    try:
      # Load and split the PDF content
      loader = PyPDFLoader(temp_file_path)
      documents = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                     chunk_overlap=0)
      texts = text_splitter.split_documents(documents)

      # Create a new vector store for each request
      vector_store = Chroma.from_documents(texts, embeddings)
      vector_store_var.set(vector_store)

      # Set up the question-answering chain with the selected model
      # qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model=selected_model),
      #                                  chain_type="stuff",
      #                                  retriever=vector_store.as_retriever())
      qa = RetrievalQA.from_chain_type(
          llm=ChatOpenAI(model=selected_model),
          chain_type="stuff",
          retriever=vector_store_var.get().as_retriever())

      # Ask the question and get the answer
      answer = qa.invoke(magic_prompt)

      return jsonify({"summary": answer}), 200
    except Exception as e:
      return jsonify({"error": str(e)}), 500
    finally:
      # Delete the temporary file
      os.unlink(temp_file_path)
  else:
    return jsonify({"error":
                    "Invalid file type. Only PDF files are allowed."}), 400


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
