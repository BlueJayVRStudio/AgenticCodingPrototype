import time

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