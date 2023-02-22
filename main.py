import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from tqdm import tqdm
from collections import Counter
import pandas as pd
from apyori import apriori

PATH = './20news-bydate-train'
K = 10

#reading all files into one dataframe
tempList = []
for target in tqdm(os.listdir(PATH)):
    for text in os.listdir(f'{PATH}/{target}'):
        with open(f'{PATH}/{target}/{text}', 'r') as f:
            tempList.append({'text': f.read()})

df = pd.DataFrame(tempList)

url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')

print('Wait for about a minute...')

#removing header
def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)
    return text

df['words'] = df['text'].apply(clean_header)

#cleaning the text
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(url, '', text)
    text = re.sub(email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    return text

df['words'] = df['words'].apply(clean_text)
df.drop('text', inplace=True, axis=1)

#removing stopwords
stop_words = stopwords.words('english')
df['words'] = df['words'].str.split() \
    .apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

#apply stemming
stemmer = PorterStemmer()
df['words'] = df['words'].str.split() \
    .apply(lambda x: ' '.join([stemmer.stem(word) for word in x]))

#get the K most frequent words for every file
def get_frequent_words(text):
    words = text.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_word_counts][0:K]

for ind in df.index:
    df['words'][ind] = get_frequent_words(df['words'][ind])

#dataframe to list
transacts = []
for ind in df.index:
    transacts.append(df['words'][ind])

#applying apriori
association_rules = apriori(transacts, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

for x in association_results:
    print(x)

#export to csv
final = pd.DataFrame(association_results)
final.to_csv('association_rules.csv', index=False)

print('The association rules are also exported in a csv file in the programm folder')
