# Implementing and Comparing Basic NLP Preprocessing
# Introduction
This script performs basic NLP preprocessing on text using NLTK, including tokenization, stopword removal, and stemming.
# Requirements
Install dependencies:
pip install nltk
Also download NLTK data:
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Code Flow
Defining the Sentence Input
A single sentence (sentence) is defined within the script.
Tokenizing the Sentence
Splits the sentence into tokens (words and punctuation) using NLTK’s word_tokenize.
Removing Stopwords
Filters out common English stopwords (e.g., “the”, “in”, “are”) using NLTK’s stopwords.
Applying Stemming
Uses NLTK’s PorterStemmer to reduce words to their root form.
# Functionality
Tokenization: Splits sentence into individual words and punctuation.
Stopword Removal: Filters out common words using nltk.corpus.stopwords.
Stemming: Uses Porter Stemmer to reduce words to root forms.
# Usage
python nlp_preprocessing.py
# output
Original Tokens – All words/punctuation split from the sentence.
Tokens Without Stopwords – Filtered to remove common English stopwords.
Stemmed Tokens – Each remaining token reduced to its root form.


# short answers
1.What is the difference between stemming and lemmatization? Provide examples with the word “running.?
Stemming:uses simple, rule-based heuristics to chop off word endings (e.g., "running" → "run" or sometimes "runn"). It does not consider the context or part of speech, so it might produce non-dictionary forms.
Lemmatization:is more sophisticated, using morphological analysis and vocabulary to convert a word to its base or dictionary form (lemma). For "running," a lemmatizer (knowing it’s a verb) would typically return "run."

2.Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?
Useful:In tasks like text classification, topic modeling, or keyword extraction, removing common words (e.g., "the", "in", "are") helps reduce noise and focuses on more meaningful tokens.
Harmful:In sentiment analysis or other tasks where subtle linguistic cues matter, removing stop words can eliminate important context. For instance, negations like "not" drastically change meaning, so discarding them can lead to incorrect interpretations.



# Named Entity Recognition with SpaCy
# Introduction
This project demonstrates Named Entity Recognition (NER) using the spaCy library. NER helps identify and classify entities such as names, dates, and locations within a text.

# Requirements
Install SpaCy and download the English model:
pip install spacy
python -m spacy download en_core_web_sm
# Code Flow
Loading the SpaCy Model:
Load SpaCy’s pre-trained English model (en_core_web_sm).
Sentence Processing:
Process the given sentence through the SpaCy pipeline.
Extracting and Printing Entities:
For each entity identified, print:
Entity text
Entity label (e.g., PERSON, DATE)
Start and end character positions in the sentence

# Code Highlights
Uses spaCy to load a pre-trained language model.
Processes an input sentence to extract:
Entity text
Entity label (e.g., PERSON, DATE)
Character positions
# Usage
Run the script in Python or Jupyter Notebook to display detected entities:
python named_entity_recognition.py
# Output
The script prints each detected entity with:
Entity text (e.g., Barack Obama)
Entity label (e.g., PERSON)
Start and end positions in the sentence

# short answers
1.How does NER differ from POS tagging in NLP?
NER (Named Entity Recognition) identifies and classifies named entities like persons, locations, organizations, and dates in text.
POS (Part-of-Speech) Tagging labels each word with its grammatical role, such as noun, verb, adjective, etc.
2.Describe two applications that use NER in the real world?
a) Financial News Analysis
Extracts company names, monetary amounts, and dates from news articles to track stock movements, earnings, and partnerships.
b) Search Engines
Enhances query understanding by recognizing entities (e.g., "Apple" as a company vs. fruit) to deliver more relevant results.





# Scaled Dot-Product Attention 
 # Introduction
 This project demonstrates how to implement the Scaled Dot-Product Attention mechanism used in Transformer models.
 # code overview
 This code demonstrates how to implement the scaled dot-product attention mechanism, a core component in transformer architectures used in NLP models. The implementation includes:
Computing the dot product between Query (Q) and Key (K) matrices.
Scaling the result by dividing by the square root of the key dimension.
Applying a softmax function to obtain attention weights.
Multiplying these weights by the Value (V) matrix to produce the final output.
# Dataset
Q = [[1, 0, 1, 0], [0, 1, 0, 1]]
K = [[1, 0, 1, 0], [0, 1, 0, 1]]
V = [[1, 2, 3, 4], [5, 6, 7, 8]]
# Code Flow
Input Matrices Definition:
Manually define Q (Query), K (Key), and V (Value).
Dot Product Calculation:
Compute the dot product of Q and the transpose of K.
Scaling:
Divide the dot product by √d, where d is the dimension of the key vectors.
Applying Softmax:
Calculate attention weights by applying softmax to the scaled scores.
Final Output Calculation:
Multiply attention weights by the Value matrix (V).
# Installation & Requirements
Ensure numpy is installed:
pip install numpy
Running the Script
python scaled_attention.py
# Output
The script will print:
Attention Weights: Matrix after applying softmax.
Final Output: Resulting matrix from multiplying attention weights by V.

# short answers
1.Why do we divide the attention score by √d in the scaled dot-product attention formula?
To prevent very large dot-product scores, which can lead to vanishing gradients during training. Scaling helps stabilize gradients and improves learning.
2.How does self-attention help the model understand relationships between words in a sentence?
Self-attention allows each word to attend to all other words in a sentence, capturing context and dependencies directly. This helps models better understand and represent relationships and semantic meanings.





# Sentiment Analysis Using HuggingFace Transformers
# Introduction
This project demonstrates the use of a pre-trained transformer model from HuggingFace’s transformers library to perform sentiment analysis on a custom input sentence.
# Requirements
Install HuggingFace Transformers:
pip install transformers
# Code Overview
Load Pipeline: A pre-trained sentiment analysis pipeline is loaded using pipeline("sentiment-analysis").
Inference: The model analyzes an input sentence and returns:
label – Sentiment class (POSITIVE or NEGATIVE)
score – Confidence score of the prediction
Output: Results are printed in a clear format
# Usage
python sentiment_transformers.py
# short answers
1.What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?
  BERT = It is a Encoder-based model and  great for understanding tasks like classification.
  GPT = It is a Decoder-based model and great for generating text or completing prompts.
