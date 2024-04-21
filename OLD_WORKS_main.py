import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

app = Flask(__name__)

# Set up the OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
# os.environ['OPENAI_API_KEY'] is already set via replit secrets

# Set up the embedding model and vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings, persist_directory=".")


@app.route('/upload', methods=['POST'])
def upload_and_process_pdf():
  if 'pdf' not in request.files:
    return jsonify({"error": "No PDF file part in the request"}), 400
  file = request.files['pdf']
  if file.filename == '':
    return jsonify({"error": "No selected file"}), 400
  if file and file.filename.endswith('.pdf'):
    save_path = os.path.join('uploads', file.filename)
    file.save(save_path)

    # Load and split the PDF content
    loader = PyPDFLoader(save_path)
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

    # Define the magic string prompt
    magic_prompt = "Summarize the key points from the uploaded PDF."

    # Ask the question and get the answer
    answer = qa.run(magic_prompt)

    return jsonify({"answer": answer}), 200
  else:
    return jsonify({"error": "Unsupported file type"}), 400


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
