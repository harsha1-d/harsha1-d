from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

#Going to create two data model (two classes) for request and response

class User(BaseModel):
    name: str
    age: int
    interest: List[str]

class UserResponse(BaseModel):
    message: str
    user: User
    recommendation:List[str]


#creating an endpoing
@app.get("/recommend")
def recommend_items(age: int, interest: str):
    """
    Example GET /recommend?age=25&interest=music
    """
    if age < 18:
        category = "Teen"
    elif age < 40:
        category = "Adult"
    else:
        category = "Senior"
        
    recommendations = []
    if interest.lower() == "music":
        recommendations = ["Spotify Playlist", "Concert Tickets"]
    elif interest.lower() == "sports":
        recommendations = ["Gym Membership", "Running Shoes"]
    elif interest.lower() == "books":
        recommendations = ["Kindle Subscription", "Bestselling Novels"]
    else:
        recommendations = ["Gift Card", "Surprise Box"]
        
    return {
        "category": category,
        "interest": interest,
        "recommendations": recommendations
            }
    
@app.post("/users", response_model=UserResponse)
def create_user(user: User):
    #Simple rule: recommend based on first interest
    first_interest = user.interest[0] if user.interest else "general"
    recommendations = []
    
    if first_interest.lower() == "music":        
        recommendations = ["Headphones", "Music App Subscription"]
    elif first_interest.lower() == "sports":
        recommendations = ["Yoga Mat", "Fitness Tracker"]
    elif first_interest.lower() == "books":
        recommendations = ["Online Library Access", "Latest Fiction Novel"]
    else:
        recommendations = ["Gift Voucher", "Surprise Item"]
        
    return {
        "message": "User profile created with recommendations",
        "user": user,
        "recommendation": recommendations
            }
    
    
    

#This is only the get method for example
# from fastapi import FastAPI

# app = FastAPI()

# #creating an endpoing
# @app.get("/recommend")
# def recommend_items(age: int, interest: str):
#     """
#     Example GET /recommend?age=25&interest=music
#     """
#     if age < 18:
#         category = "Teen"
#     elif age < 40:
#         category = "Adult"
#     else:
#         category = "Senior"
        
#     recommendations = []
#     if interest.lower() == "music":
#         recommendations = ["Spotify Playlist", "Concert Tickets"]
#     elif interest.lower() == "sports":
#         recommendations = ["Gym Membership", "Running Shoes"]
#     elif interest.lower() == "books":
#         recommendations = ["Kindle Subscription", "Bestselling Novels"]
#     else:
#         recommendations = ["Gift Card", "Surprise Box"]
        
#     return {
#         "category": category,
#         "interest": interest,
#         "recommendations": recommendations
#             }
    
    
#     #http://127.0.0.1:8000/recommend?age=23&interest=%22music%22    #http://127.0.0.1:8000/recommend?age=23&interest=%22music%22