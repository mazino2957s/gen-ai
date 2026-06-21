import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")
seed_word = "technology"
similar_words = model.most_similar(seed_word, topn=5)
words = [word for word, score in similar_words]
paragraph = f"{seed_word} is growing fast. It includes {words[0]}, {words[1]}, and {words[2]}. These help in {words[3]} and {words[4]}."
print("Similar words:", words)
print("Paragraph:", paragraph)
