import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

import chardet

# Detect encoding for positive words
with open('positive-words.txt', 'rb') as f:
    result_pos = chardet.detect(f.read())

# Detect encoding for negative words
with open('negative-words.txt', 'rb') as f:
    result_neg = chardet.detect(f.read())

# Open files with detected encodings
positive_words = set(open('positive-words.txt', encoding=result_pos['encoding']).read().split())
negative_words = set(open('negative-words.txt', encoding=result_neg['encoding']).read().split())



df = pd.read_excel('Input.xlsx')

# Define functions for text analysis
def extract_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1').get_text()
    paragraphs = soup.find_all('p')
    article_text = ' '.join([para.get_text() for para in paragraphs])
    return title, article_text

def positive_score(text):
    words = word_tokenize(text.lower())
    positive_count = sum(1 for word in words if word in positive_words)
    return positive_count

def negative_score(text):
    words = word_tokenize(text.lower())
    negative_count = sum(1 for word in words if word in negative_words)
    return negative_count

def polarity_score(pos_score, neg_score):
    return (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)

def subjectivity_score(pos_score, neg_score, total_words):
    return (pos_score + neg_score) / (total_words + 0.000001)

def avg_sentence_length(text):
    sentences = sent_tokenize(text)
    total_words = len(word_tokenize(text))
    return total_words / len(sentences)

def count_syllables(word):
    syllables = 0
    vowels = "aeiou"
    word = word.lower()
    if word[0] in vowels:
        syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllables += 1
    if word.endswith("e"):
        syllables -= 1
    if syllables == 0:
        syllables += 1
    return syllables

def is_complex_word(word):
    return count_syllables(word) >= 3

def percentage_of_complex_words(text):
    words = word_tokenize(text)
    complex_words = sum(1 for word in words if is_complex_word(word))
    return complex_words / len(words)

def fog_index(text):
    asl = avg_sentence_length(text)
    pcw = percentage_of_complex_words(text)
    return 0.4 * (asl + pcw)

def avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    total_words = len(word_tokenize(text))
    return total_words / len(sentences)

def complex_word_count(text):
    words = word_tokenize(text)
    return sum(1 for word in words if is_complex_word(word))

def word_count(text):
    words = word_tokenize(text)
    return len(words)

def syllables_per_word(text):
    words = word_tokenize(text)
    total_syllables = sum(count_syllables(word) for word in words)
    return total_syllables / len(words)

def personal_pronouns(text):
    words = word_tokenize(text)
    pronouns = ['i', 'we', 'my', 'ours', 'us']
    pronoun_count = sum(1 for word in words if word.lower() in pronouns)
    return pronoun_count

def avg_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    return total_characters / len(words)

# Process each URL and compute the required variables
results = []
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    title, article_text = extract_article_text(url)
    
    pos_score = positive_score(article_text)
    neg_score = negative_score(article_text)
    pol_score = polarity_score(pos_score, neg_score)
    subj_score = subjectivity_score(pos_score, neg_score, word_count(article_text))
    asl = avg_sentence_length(article_text)
    pcw = percentage_of_complex_words(article_text)
    fog_idx = fog_index(article_text)
    avg_words_sent = avg_words_per_sentence(article_text)
    complex_wc = complex_word_count(article_text)
    wc = word_count(article_text)
    spw = syllables_per_word(article_text)
    pp = personal_pronouns(article_text)
    awl = avg_word_length(article_text)
    
    results.append([url_id, url, pos_score, neg_score, pol_score, subj_score, asl, pcw, fog_idx, avg_words_sent, complex_wc, wc, spw, pp, awl])

# Convert results to DataFrame and save to Excel
output_df = pd.DataFrame(results, columns=[
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
    'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
    'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
])

output_df.to_excel('Output Data Structure.xlsx', index=False)
