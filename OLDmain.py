import os
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_pdf():
  if 'pdf' not in request.files:
    return jsonify({"error": "No PDF file part in the request"}), 400
  file = request.files['pdf']
  if file.filename == '':
    return jsonify({"error": "No selected file"}), 400
  if file and file.filename.endswith('.pdf'):
    save_path = os.path.join('uploads', file.filename)
    file.save(save_path)
    # Placeholder: Here you can add your chromaDB and OpenAI API code
    return jsonify({
        "message": "File uploaded successfully",
        "path": save_path
    }), 200
  else:
    return jsonify({"error": "Unsupported file type"}), 400


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
