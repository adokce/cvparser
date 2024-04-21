import os
import tempfile
import logging
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
# os.environ['OPENAI_API_KEY'] is already set via replit secrets

# Set up the embedding model and vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings, persist_directory=".")

# Read the magic prompt from the file
with open("prompt.txt", "r") as file:
  magic_prompt = file.read().strip()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    if 'pdf' not in request.files:
      return render_template('index.html',
                             error="No PDF file found in the request")
    file = request.files['pdf']
    if file.filename == '':
      return render_template('index.html', error="No file selected")

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

        # Create a vector store from the PDF content
        vector_store.add_documents(texts)

        # Set up the question-answering chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                         chain_type="stuff",
                                         retriever=vector_store.as_retriever())

        # Ask the question and get the answer
        answer = qa.run(magic_prompt)

        return render_template('index.html', summary=answer)
      except Exception as e:
        return render_template('index.html', error=str(e))
    else:
      return render_template(
          'index.html', error="Invalid file type. Only PDF files are allowed.")

  return render_template('index.html')


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
