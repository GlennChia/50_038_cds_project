'''
1. Numbers
2. Apostrophe
3. All punctuations
4. Weird symbols
5. Stop words
'''

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction import stop_words
sets=[stop_words.ENGLISH_STOP_WORDS]
sklearnStopWords = [list(x) for x in sets][0]
token=ToktokTokenizer()
stopWordList=stopwords.words('english')
stopWords = stopWordList + sklearnStopWords
stopWords = list(dict.fromkeys(stopWords))

file_name = 'TED_Transcripts_short.csv'
df = pd.read_csv('../owentemple-ted-talks-complete-list/{}'.format(file_name))
df = df.dropna(subset=['transcript'])
df = df.reset_index(drop=True)

def stopWordsRemove(text):
    wordList=[x.lower().strip() for x in token.tokenize(text)]
    removedList=[x + ' ' for x in wordList if not x in stopWords]
    text=''.join(removedList)
    return text
​
# There is a mispelt word that needs to be replaced
df['transcript'] = df['transcript'].str.replace('childrn','children')

df['transcript'] = df['transcript'].str.replace('\r',' ')
df['transcript'] = df['transcript'].str.replace("\'s"," is")
df['transcript'] = df['transcript'].str.replace("\'m"," am")
df['transcript'] = df['transcript'].str.replace("\'ll"," will")
df['transcript'] = df['transcript'].str.replace("n\'t"," not")
df['transcript'] = df['transcript'].str.replace("\'ve"," have")
df['transcript'] = df['transcript'].str.replace("\'re"," are")
df['transcript'] = df['transcript'].str.replace("\'d"," would")
df['transcript'] = df['transcript'].str.replace(r"\(([^)]+)\)","")
# Deal with Mr. and Dr.
df['transcript'] = df['transcript'].str.replace("mr.","mr")
df['transcript'] = df['transcript'].str.replace("Mr.","mr")
df['transcript'] = df['transcript'].str.replace("dr.","dr")
df['transcript'] = df['transcript'].str.replace("mrs.","mrs")
df['transcript'] = df['transcript'].str.replace("Mrs.","mrs")
df['transcript'] = df['transcript'].str.replace("Dr.","dr")

df['transcript'] = df['transcript'].str.replace(r'\d+','')
df['transcript'] = df['transcript'].str.replace(r'<.*?>','')
for i in string.punctuation:
    if i == "'":
        df['transcript'] = df['transcript'].str.replace(i,'')
    else:
        df['transcript'] = df['transcript'].str.replace(i,' ')
df['transcript'] = df['transcript'].str.replace('\s+',' ')
df['transcript'] = df['transcript'].map(lambda com : stopWordsRemove(com))