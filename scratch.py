from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
try:
    print("chunk_overlap:", splitter.chunk_overlap)
except Exception as e:
    print("Error:", e)

try:
    print("_chunk_overlap:", splitter._chunk_overlap)
except Exception as e:
    print("Error:", e)
