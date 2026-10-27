# Exercise 4.18
# Modify the previous example Python code (Example 4.22) and choose a different piece of text
# for text analysis. Comment on the results.
#
# Comment on results:
# NLTK word_tokenize splits the new sentence into words and punctuation tokens.
# pos_tag assigns part-of-speech tags (such as NNP for proper nouns like 'Deep', 'Learning',
# 'Yann', 'LeCun', 'Google', VBZ/VBD for verbs, JJ for adjectives).
# ne_chunk groups recognized named entities (such as PERSON for 'Yann LeCun' and
# ORGANIZATION for 'Google') into tree chunks.

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

sentence = """
Deep learning has revolutionized computer vision and natural language processing.
Pioneers like Yann LeCun, Geoffrey Hinton, and Yoshua Bengio developed foundational
neural network architectures at universities and companies such as Google and Meta
in North America and Europe.
"""

print("=== 1. Tokens ===")
tokens = nltk.word_tokenize(sentence)
print(tokens)

print("\n=== 2. POS Tagged Tokens ===")
tagged = nltk.pos_tag(tokens)
print(tagged)

print("\n=== 3. Named Entities ===")
entities = nltk.chunk.ne_chunk(tagged)
print(entities)
