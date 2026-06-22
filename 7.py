from transformers import pipeline
print("loading modules")
summarizer = pipeline("summarization")
print("Model loaded successfully.\n")
text = """
The Industrial Revolution, which took place from the 18th to the 19th centuries, 
transformed agrarian societies into industrial and urban societies. Factories, 
machinery, and mass production improved manufacturing efficiency. However, 
industrialization also created harsh working conditions for many people.
"""
summary = summarizer(text)
print(summary)

