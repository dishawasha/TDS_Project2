import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
# Import the solver function we will write next
from solver import solve_quiz


import os
from dotenv import load_dotenv

# Load the environment variables from .env
load_dotenv() 

# Now, retrieve the variable from the environment.
# Since the default value is in .env, you can remove the default from the code.
# If you still want a *code-level* default fallback if the key is missing from .env/env, keep the default.

# Option A: Rely purely on the environment (Recommended clean method)
MY_SECRET = os.getenv("MY_SECRET")

# Option B: Keep the code-level default as a backup (Closer to your original line)
# MY_SECRET = os.getenv("MY_SECRET", "apple")

#print(f"The secret is: {MY_SECRET}")


# Define the expected input JSON structure
class QuizTask(BaseModel):
   email: str
   secret: str
   url: str


app = FastAPI(title="TDS Automated Quiz Solver")


@app.post("/quiz-endpoint/")
def receive_task(task: QuizTask):
   # 1. Verification Logic (HTTP 403 check)
   if task.secret != MY_SECRET:
       raise HTTPException(status_code=403, detail="Invalid secret provided.")


   # 2. Start the solver agent
   # NOTE: For a production app, you'd run solve_quiz in a background thread
   # to ensure the HTTP 200 response is *immediate*. For simplicity now,
   # we run it synchronously.
   try:
       solve_quiz(task.email, task.secret, task.url)
       # 3. HTTP 200 Response
       return {"status": "Solving process started successfully.", "quiz_url": task.url}
   except Exception as e:
       # Handle solver errors
       return {"status": "Solver failed to execute.", "error": str(e)}


# Reminder to run: uvicorn main:app --reload --port 8000

