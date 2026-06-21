from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
sentences = [
    "I love learning artificial intelligence.",

    "This movie was terrible and boring.",

    "The product is okay, not too bad.",

    "I am extremely happy with the results.",

    "The service was disappointing."
]
results = sentiment(sentences)
print(results)
