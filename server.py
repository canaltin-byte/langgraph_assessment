import uvicorn
import os

# Ensure templates directory exists
os.makedirs("templates", exist_ok=True)

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True) 