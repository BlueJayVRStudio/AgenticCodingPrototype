from core.SingleAgent import SingleAgent

singleAgent = SingleAgent()

while True:
    # print(singleAgent.messages)
    result, success = singleAgent.run(input("Ask: "))
    if success:
        print(result)
        print("TASK COMPLETED")
    else:
        print(result)
        print("CONVERSATION COMPLETE")

