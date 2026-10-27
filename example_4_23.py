# Example 4.23 (Transformers.py)
# https://github.com/huggingface/transformers
# pip install transformers

from transformers import pipeline

classifier = pipeline('sentiment-analysis', framework='pt')
result = classifier('This is a good movie.')
print(result)
