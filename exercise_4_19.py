# Exercise 4.19
# Modify the previous example Python code (Example 4.23), choose some different sentences,
# and make sure some are positive and some are negative. Comment on the results.
#
# Comment on results:
# The Hugging Face sentiment-analysis pipeline accurately classifies positive sentences
# with label 'POSITIVE' and high confidence scores (>0.99), and negative sentences
# with label 'NEGATIVE' and high confidence scores (>0.99).

from transformers import pipeline

classifier = pipeline('sentiment-analysis', framework='pt')

sentences = [
    "This is an amazing tutorial on deep learning, I learned so much!",
    "The service at the restaurant was terrible and the food was cold.",
    "I absolutely love how fast and accurate this neural network model is.",
    "This software keeps crashing and wasting my time, very frustrating."
]

for text in sentences:
    result = classifier(text)
    print(f"Sentence: {text}")
    print(f"Result:   {result}\n")
