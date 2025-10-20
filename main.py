from agents.base_agent import BaseAgent

singleAgent = BaseAgent()

while True:
    # print(singleAgent.messages)
    result, success = singleAgent.run(input("Ask: "))
    if success:
        print(result)
        print("TASK COMPLETED")
    else:
        print(result)
        print("CONVERSATION COMPLETE")

