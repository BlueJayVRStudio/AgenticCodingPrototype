import shutil, os
import pytest
from agents.base_agent import BaseAgent
import time
import random

def safe_run_agent(agent, prompt):
    time.sleep(2)
    for attempt in range(5):
        try:
            return agent.run(prompt)
        except Exception as e:
            if "rate limit" in str(e).lower():
                sleep_time = (2 ** attempt) + random.random()
                print(f"Rate limited, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                raise

@pytest.fixture(scope="function", autouse=True)
def clean_test_docs():
    path = "test_docs"
    if os.path.exists(path):
        shutil.rmtree(path)
    yield
    if os.path.exists(path):
        shutil.rmtree(path)

@pytest.mark.integration
def test_base_agent_integrations():
    agent = BaseAgent("test_agent")
    result, success = safe_run_agent(agent, "Can you sum up 4 and 5?")
    assert result[0] == "The sum between 4 and 5 is: 9", "failed to add two numbers"
    assert success
    
    result, success = safe_run_agent(agent, "Can you create a directory called dir_from_pytest")
    assert os.path.exists("./test_docs/dir_from_pytest"), "directory creation failed"

    result, success = safe_run_agent(agent, (
        "Please ignore the previous directory created. Now "
        "Can you create a directory called some_python_scripts "
        "then under that directory create two files. One being "
        "a python hello world script with just a single print "
        "statement of 'hello world!' and the other being a text "
        "file of a grocery list that lists apple and pineapple."
    ))
    print(result)
    assert success
    assert len(result) == 3
    assert os.path.exists("./test_docs/some_python_scripts"), "directory creation failed"
    assert os.path.exists("./test_docs/some_python_scripts/hello_world.py"), "script creation failed"
    assert os.path.exists("./test_docs/some_python_scripts/grocery_list.txt"), "text creation failed"
    with open("./test_docs/some_python_scripts/grocery_list.txt", "r") as f:
        for line in f:
            assert line.strip() in ["apple", "pineapple"]

    result, success = safe_run_agent(agent, "Let's have a conversation. Can you give a short summary of what we did today?")
    assert not success
    assert len(result) == 1
    assert type(result) == list
    