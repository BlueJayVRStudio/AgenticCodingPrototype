from core.factory.agent_factory import AgentFactory

agent_factory = AgentFactory()
singleAgent = agent_factory.create_base_agent("base_agent")

while True:
    # print(singleAgent.messages)
    result, success = singleAgent.run(input("Ask: "))
    if success:
        print(result)
        print("TASK COMPLETED")
    else:
        print(result)
        print("CONVERSATION COMPLETE")

