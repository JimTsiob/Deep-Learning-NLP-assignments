from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

greek_text = "Καλημέρα! Ονομάζομαι Μαρία."

# Tokenize the text into words
tokenized_text = word_tokenize(greek_text)

# Create Word2Vec model
model = Word2Vec(sentences=[tokenized_text], vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model.save("greek_word2vec.model")

# Load the model
# model = Word2Vec.load("greek_word2vec.model")

# Get the vector for a word
vector = model.wv['Καλημέρα']
print("Vector for 'Καλημέρα':", vector)