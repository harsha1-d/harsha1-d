from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI() #app is an instance of FastAPI class which is used to create the API.

API_KEY = "12345ABCDEF"

@app.middleware("http")
async def check_api_key(request, call_next):
    key = request.headers.get("X-API-KEY")
    if key != API_KEY:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)

#Root the endpoint
@app.get("/Welcome") # To write a Get or Post API we need to use the decorator which is provided by FastAPI. 
#Here we are using @app.get() decorator to create a Get API. 
#The parameter of the decorator is the endpoint of the API.
#Where do we write this @app annotation? We write this annotation above the function which we want to execute when the API is called by someone. Below is the function which will be executed when the API is called. The name of the function can be anything. Here I have named it welcome() but you can name it anything you want. The return statement of the function is the response which will be sent to the client when the API is called. Here we are returning a dictionary with a key "message" and a value "welcome to my FastAPI application!".
def welcome():
    return {"message": "welcome to my FastAPI application!"}

#GET API with a parameter
@app.get("/user")
def user_profile():
    return {"name": "Sanjay Kumar", "Channel": "Applied With AI - Tamil",
            "Website": "https://www.appliedwithai.com/", "LinkedIN":"www.linkedin.com/in/harshavardhan-dharman-a062b0240"}
    
@app.get("/user/{user_id}") #app.get("/user/{-----}") -> Here the content inside the {} is the path parameter. This means that when we call this API, we need to provide a value for this path parameter. For example, if we want to call this API with a user_id of 1, we need to call it like this: /user/1. The value of the path parameter will be passed to the function as an argument. Below is the function which will be executed when the API is called with a user_id. The name of the function can be anything. Here I have named it user_profile() but you can name it anything you want. The return statement of the function is the response which will be sent to the client when the API is called. Here we are returning an empty dictionary but you can return any data you want.
def user_profile(user_id:int):
    if(user_id == 1):
        return {
            "name": "Harshavardhan Dharman", "Channel": "Applied With AI - Tamil",
            "Website": "https://www.appliedwithai.com/", "LinkedIN":"www.linkedin.com/in/harshavardhan-dharman-a062b0240"
        }
    else:
        return {
            "name": "Mano"+str(user_id), "Channel": "Applied With AI - Tamil",
            "Website": "https://www.appliedwithai.com/", "LinkedIN":"www.linkedin.com/in/harshavardhan-dharman-a062b0240"
        }
        
        
#How to build a post API in FastAPI?
#To build a post API in FastAPI, we need to use the @app.post().
#POST API is not straight forward as GET API because in POST API we need to send some data to the server. This data can be sent in the form of JSON, form data, or any other format. In FastAPI, we can use Pydantic models to define the structure of the data which we want to receive in the POST API. Below is an example of how to create a POST API in FastAPI using Pydantic models.
#So here we are going to get the name, age, email as an input from the user and we will return the same data as a response. Below is the code for the POST API.
#Below is the input from the user which we want to receive in the POST API. We will create a Pydantic model to define the structure of this data.
{
    "name": "Harshavardhan Dharman",
    "age": 25,
    "email": "harshavardhan.dharman@example.com"
}


class User(BaseModel): #Here we are creating a Pydantic model named User which inherits from BaseModel. The name of the class can be anything. Here I have named it User but you can name it anything you want. The attributes of the class are the fields which we want to receive in the POST API. Here we have three fields: name, age, and email. The type of each field is specified after the colon. Here we have specified the type of name as str, age as int, and email as str.
    name: str
    age: int
    email: str
    
users=[]

@app.post("/users")
def create_user(user: User): #Here we are creating a POST API with the endpoint /users. The function which will be executed when this API is called is create_user(). The name of the function can be anything. Here I have named it create_user() but you can name it anything you want. The parameter of the function is user which is of type User. This means that when we call this API, we need to send a JSON object which has the same structure as the User model. The return statement of the function is the response which will be sent to the client when the API is called. Here we are returning the same data which we received in the request body as a response.
    users.append(user.dict())
    return {"message": "User created successfully!", "total_users": len(users)}