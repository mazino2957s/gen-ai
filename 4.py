import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")

prompt = "Explain artificial intelligence"

similar = model.most_similar("artificial", topn=3)
words = [w for w, s in similar]

new_prompt = prompt + " including " + ", ".join(words)

print("Original:", prompt)
print("Improved:", new_prompt)
