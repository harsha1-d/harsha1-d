from pydantic import BaseModel, EmailStr

class Person(BaseModel):
    name: str
    age: int
    email: EmailStr
    
valid_data = Person(name="John", age=30, email="John@gmail.com")
print(valid_data)