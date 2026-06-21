from gensim.models import Word2Vec

sentences = [
    ["king", "is", "strong"],
    ["queen", "is", "powerful"],
    ["man", "is", "human"],
    ["woman", "is", "human"],
    ["prince", "is", "boy"],
    ["princess", "is", "girl"]
]

model = Word2Vec(sentences, vector_size=50, window=2, min_count=1)

print("Vector of 'king':")
print(model.wv['king'])

print("\nSimilar words to 'king':")
print(model.wv.most_similar('king'))

print("\nking - man + woman = ?")
print(model.wv.most_similar(positive=['king', 'woman'], negative=['man']))
