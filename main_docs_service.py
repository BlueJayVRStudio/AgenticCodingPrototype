from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from core.factory.agent_factory import AgentFactory
import json

# Initialize agent
agent_factory = AgentFactory()
document_checker_agent = agent_factory.create_document_checker_agent()

app = FastAPI()

class CheckRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    verdict: bool
    suggested_edit: str

@app.post("/check", response_model=CheckResponse)
def check_document(payload: CheckRequest):
    """
    TODO: lock agent while executing query
    """

    # Run the agent
    raw_result = document_checker_agent.run(payload.text)

    # Parse JSON coming back from the model
    result = json.loads(raw_result)

    return result

if __name__ == "__main__":
    uvicorn.run("main_docs_service:app", host="localhost", port=8500, reload=True)
