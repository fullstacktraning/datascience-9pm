# http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/users/101
# http://127.0.0.1:8000/search?q=Hello
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
@app.get("/")
def home():
    return {"msg":"welcome to get request"}

@app.get("/users/{user_id}")
def get_user(user_id:int):
    return {"user_id":user_id}

@app.get("/search")
def search(q:str):
    return {"messag":q}

class Login(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(login:Login):
    if login.username == "admin" and login.password=="admin@123":
        return {"login":"success"}
    else:
        return {"login":"fail"}
    
@app.put("/update")
def update_user():
    return {"msg":"welcome to basic put request !!!"}

@app.delete("/delete-user")
def delete_user():
    return {"msg":"welcome to basic delete request !!!"}