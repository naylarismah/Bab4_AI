# Exercise 4.20
# Modify the previous example Python code (Example 4.24), choose a different piece of text,
# and ask different questions. Comment on the results.
#
# Comment on results:
# The question-answering pipeline extracts the exact span of text from the given context
# that answers each question, along with start/end character indices and confidence scores.

from transformers import pipeline

question_answerer = pipeline('question-answering', framework='pt')

context = (
    "AlexNet is a convolutional neural network developed by Alex Krizhevsky, "
    "Ilya Sutskever, and Geoffrey Hinton at the University of Toronto. "
    "In 2012, AlexNet won the ImageNet Large Scale Visual Recognition Challenge "
    "with a top-five accuracy of 85 percent."
)

questions = [
    "Who developed AlexNet?",
    "Where was AlexNet developed?",
    "When did AlexNet win the ImageNet challenge?",
    "What was the top-five accuracy of AlexNet?"
]

for q in questions:
    result = question_answerer({
        'question': q,
        'context': context
    })
    print(f"Question: {q}")
    print(f"Answer:   {result}\n")
