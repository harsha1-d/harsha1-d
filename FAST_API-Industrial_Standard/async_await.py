import httpx
import time
import asyncio
from fastapi import FastAPI

app = FastAPI()

JOKE_URL = "https://official-joke-api.appspot.com/jokes/random"

@app.get("/jokes-sync")
def get_jokes_sync():
    start = time.time()
    jokes = []
    with httpx.Client() as client: #httpx.Client() is used to access any publically availabele end-point or URL
        for _ in range(10):
            resp = client.get(JOKE_URL)
            data = resp.json()
            jokes.append(f"{data['setup']} - {data['punchline']}")
    elapsed = time.time() - start
    
    return {
        "mode": "sync",
        "jokes": jokes,
        "elapsed_time_sec": round(elapsed,3)
    }  
#Here we are declaring async every where because this makes the process asynchronous. If we do not declare async, then the process will be synchronous and the total time taken will be the sum of the time taken for each request. In contrast, asynchronous code allows multiple requests to be made concurrently, which can significantly reduce the total time taken when making multiple requests.
@app.get("/jokes-async")
async def get_jokes_async():
    start = time.time()
    jokes = []
    async with httpx.AsyncClient() as client:
        tasks = [client.get(JOKE_URL) for _ in range(10)]
        responses = await asyncio.gather(*tasks) #asyncio.gather is used to run multiple asynchronous tasks concurrently and wait for all of them to complete. It takes in a list of tasks (in this case, the list of GET requests) and returns a list of results once all tasks have completed. The await keyword is used to pause the execution of the function until all the tasks have finished, allowing other operations to run concurrently in the meantime.
                                                 #Also the * operator is used to unpack the list of tasks into individual arguments for the asyncio.gather function. This allows us to pass each task as a separate argument to the function, rather than passing the entire list as a single argument. By using asyncio.gather with the * operator, we can efficiently handle multiple asynchronous requests concurrently and process their results once they are all completed.      
        for resp in responses:
            data = resp.json()
            jokes.append(f"{data['setup']} - {data['punchline']}")
    elapsed = time.time() - start
    return {
        "mode": "async",
        "jokes": jokes,
        "elapsed_time_sec": round(elapsed,3)
    }
    
    
    
#This is synchronous code, which means that each request is made one after the other, and the total time taken will be the sum of the time taken for each request. In contrast, asynchronous code allows multiple requests to be made concurrently, which can significantly reduce the total time taken when making multiple requests.
# import httpx
# import time
# from fastapi import FastAPI

# app = FastAPI()

# JOKE_URL = "https://official-joke-api.appspot.com/jokes/random"

# @app.get("/jokes-sync")
# def get_jokes_sync():
#     start = time.time()
#     jokes = []
#     with httpx.Client() as client: #httpx.Client() is used to access any publically availabele end-point or URL
#         for _ in range(10):
#             resp = client.get(JOKE_URL)
#             data = resp.json()
#             jokes.append(f"{data['setup']} - {data['punchline']}")
#     elapsed = time.time() - start

#     return {
#         "mode": "sync",
#         "jokes": jokes,
#         "elapsed_time_sec": round(elapsed,3)
#     }