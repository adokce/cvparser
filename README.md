```mermaid
flowchart TB
User(("User Uploads PDF & Inputs Query")) --> PDFExtraction[("Extract PDF Text\n(PyPDFLoader)")]
PDFExtraction --> SegmentText[("Segment Text\n(RecursiveCharacterTextSplitter)")]
SegmentText --> GenerateEmbeddings[("Generate Text Embeddings\n(OpenAIEmbeddings)")]
GenerateEmbeddings --> StoreEmbeddings[("Store & Retrieve Embeddings\n(Chroma)")]
StoreEmbeddings --> ProcessQuery[("Process Query Using LLM\n(RetrievalQA, Selected Model)")]
ProcessQuery --> Display[("Display Answer to User")]
```

### `/upload` Endpoint

**Description**: This endpoint handles the uploading and processing of PDF files. It extracts text from the uploaded PDF, processes it with the specified AI model, and generates a response based on the provided query (magic prompt).

**URL**: `/upload`

**Method**: `POST`

**Content-Type**: `multipart/form-data`

**Request Body**:
- **`pdf`** (required, file): The PDF file to be uploaded and processed.
- **`model`** (optional, string): Specifies the AI model to use for processing the text. Default is `gpt-3.5-turbo`.
  - Options:
    - `gpt-3.5-turbo`
    - `gpt-4-turbo-2024-04-09`
    - `meta/meta-llama-3-8b-instruct`
- **`magic_prompt`** (optional, string): A query or command describing what to do with the text. Default is extracted from `prompt.txt`.


<details>
<summary>Responses</summary>

**Success Response**:
- **Code**: `200 OK`
- **Content**: 
  ```json
  {
    "summary": "Extracted answer based on the provided prompt."
  }
  ```

**Error Responses**:

- **Code**: `400 Bad Request`
- **Content**: 
  ```json
  {
    "error": "No PDF file found in the request"
  }
  ```
- **Code**: `400 Bad Request`
- **Content**: 
  ```json
  {
    "error": "No file selected"
  }
- **Code**: `500 Internal Server Error`
- **Content**: 
  ```json
  {
    "error": "An unexpected error occurred"
  }
</details>

**Example**:
```
curl -X POST http://localhost:8080/upload \
     -F "pdf=@path_to_your_document.pdf" \
     -F "model=gpt-3.5-turbo" \
     -F "magic_prompt=Summarize the key points of this document"
```