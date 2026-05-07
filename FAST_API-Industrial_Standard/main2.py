from fastapi import FastAPI, Body, Form, UploadFile, File
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool

#Handling the JSON body in POST request
@app.post("/json")
def receive_json(item: Item):
    return {
        "type": "JSON",
        "name": item.name,
        "price": item.price,
        "in_stock": item.in_stock
    }
    
#Handling the TEXT body in POST request
@app.post("/text")
def receive_text(content: str = Body(..., media_type="text/plain")):  #Here the media_type is set to text/plain to indicate that we expect plain text in the request body. By default FastAPI expects JSON, so we need to specify this to handle plain text correctly. Similarly the "..." indicates that this field is required and must be provided with something.
    return {
        "type": "Plain Text",
        "content": content
    }
    
#Handling the FORM body in POST request - Used to get user name and password from the form
@app.post("/form")
def receive_form(username: str = Form(...), password: str = Form(...)):
    return {
        "type": "Form Data",
        "username": username,
        "password": password
    }
    
#Handling the FILE body in POST request - Used to upload a file. Usually in RAG applications we use this to upload documents(PDF's) for processing and extracting information from them. This is a common use case in RAG applications where we need to process and extract information from uploaded documents, such as PDFs, to provide relevant responses or insights based on the content of those documents.
@app.post("/upload")
def receive_file(file: UploadFile = File(...)):
    return {
        "type": "File Upload",
        "filename": file.filename,
        "content_type": file.content_type
    }