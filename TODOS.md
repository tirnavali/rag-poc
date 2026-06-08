# embedding output dimension should be least possible
256 altına düşmemeli 1024 milyonlarca veride maliyete sebep olur.
# late chunking
late chunking 512 input dimensonda anlamsız fakat default open olamalı 
Overlap size %1~2 arası idealdir.
# vector store text size belirme
Minimum 384 token olacak şeklide bir optimizasyon yapalım.
# docling HybridChunking incelenecek

# daha iyi bir llm besleme örneği tutaklar için
Better prompts: At answer time you can tell the LLM: “You’re seeing an excerpt spoken by {speaker}, on {date}, during {session}.” This helps it generate accurate attributions.
# rate limiting per user
Bu durum mcp yapısına geçince ele alınmalı