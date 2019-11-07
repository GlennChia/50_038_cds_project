'''
1. Numbers
2. Apostrophe
3. Punctuatations other than full stops (In case we want to do paragraphs)
4. Weird symbols
'''

import pandas as pd
import string

file_name = 'TED_Transcripts_short.csv'
df = pd.read_csv('../owentemple-ted-talks-complete-list/{}'.format(file_name))
df = df.dropna(subset=['transcript'])
df = df.reset_index(drop=True)

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
    elif i == ".":
        continue
    else:
        df['transcript'] = df['transcript'].str.replace(i,' ')
df['transcript'] = df['transcript'].str.replace('\s+',' ')
