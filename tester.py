from langchain.memory import ConversationBufferMemory

memory1 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory1.save_context({"input": "hi"}, {"output": "whats up"})
memory1.chat_memory.add_user_message("hi2")
memory1.chat_memory.add_ai_message("what's up2?")

memory2 = ConversationBufferMemory()
memory2.save_context({"input": "hi"}, {"output": "whats up"})
memory2.chat_memory.add_user_message("hi2")
memory2.chat_memory.add_ai_message("what's up2?")


print(memory1.load_memory_variables({}))
print(memory2.load_memory_variables({}))
