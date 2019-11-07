#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import seaborn as sns
from bokeh.io import output_notebook, show
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models.tools import HoverTool
from bokeh.models import ColumnDataSource, CustomJS
import string
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction import stop_words
from nltk.stem.wordnet import WordNetLemmatizer

sets=[stop_words.ENGLISH_STOP_WORDS]
sklearnStopWords = [list(x) for x in sets][0]
token=ToktokTokenizer()
lemma=WordNetLemmatizer()
stopWordList=stopwords.words('english')
stopWords = stopWordList + sklearnStopWords
stopWords = list(dict.fromkeys(stopWords))
# import gensim

st.title('TranScribe')

st.title("Visualizations")

DATA_DIR = "../owentemple-ted-talks-complete-list"

data = DATA_DIR + '/data/ted_talks_by_id_plus_transcripts_and_liwc_and_mft_plus_views.csv'

def stopWordsRemove(text):
    wordList=[x.lower().strip() for x in token.tokenize(text)]
    removedList=[x + ' ' for x in wordList if not x in stopWords]
    text=''.join(removedList)
    return text


def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w,'v')
        listLemma.append(x)
    return text


@st.cache
def load_data(filename):
    df = pd.read_csv(filename, index_col="id", parse_dates=['date_published'])
    df = df.dropna(subset=['transcript'])
    df['duration'] = pd.to_timedelta(df['duration'])
    # There is a mispelt word that needs to be replaced
    df['cleaned_transcript'] = df['transcript']
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace('\r',' ')
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("\'s"," is")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("\'m"," am")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("\'ll"," will")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("Can\'t","cannot")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("Sha\'t","shall not")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("Won\'t","would not")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("n\'t"," not")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("\'ve"," have")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("\'re"," are")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("\'d"," would")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace(r"\(([^)]+)\)","")
    # Deal with Mr. and Dr.
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("mr. ","mr ")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("Mr. ","m ")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("mrs. ","mrs ")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("Mrs. ","mrs ")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("Dr. ","dr ")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace("dr. ","dr ")
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace(r'\d+','')
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace(r'<.*?>','')
    for i in string.punctuation:
        if i == "'":
            df['cleaned_transcript'] = df['cleaned_transcript'].str.replace(i,'')
        else:
            df['cleaned_transcript'] = df['cleaned_transcript'].str.replace(i,' ')
    df['cleaned_transcript'] = df['cleaned_transcript'].str.replace('\s+',' ')
    
    df['lemmatized_transcript'] = df['cleaned_transcript']
    df['lemmatized_transcript'] = df['lemmatized_transcript'].map(lambda com : stopWordsRemove(com))
    df['lemmatized_transcript'] = df['lemmatized_transcript'].map(lambda com : lemitizeWords(com))
    df['lemmatized_transcript'] = df['lemmatized_transcript'].str.replace('\s+',' ')

    return df


ted_talks = load_data(data)

# Process tags
tags = ted_talks['tags'].str.replace(', ', ',').str.lower().str.strip()
split_tags = tags.str.split(',')
tag_counts_per_talk = split_tags.apply(len)

joined_tags = tags.str.cat(sep=',').split(',')
all_tags = pd.Series(joined_tags)

tag_counts = all_tags.value_counts()

cumulative_tag_counts = np.cumsum(tag_counts)
cumulative_tag_counts.index = range(len(cumulative_tag_counts))
tag_ratios = cumulative_tag_counts / sum(tag_counts)

# process duration
talk_time_minutes = ted_talks['duration'].dt.total_seconds() / 60
word_count = ted_talks['transcript'].str.split().apply(len).value_counts()
word_count = word_count.to_frame()
word_count = word_count.reset_index()
word_count = word_count.rename(columns={"index": "word_counts", 'transcript': "num_docs"})
total_words = word_count['word_counts'].sum()


## Begin plotting

index = list(range(len(tag_counts)))
counts = list(tag_counts.values)
source = ColumnDataSource(data=dict(index=index, counts=counts))

p = figure(plot_height=400,
           plot_width=600,
           title='Frequency of tags in documents')

p.line(x='index', y='counts',
       line_width=2,
       color='mediumvioletred',
       source=source)

renderer = p.circle(x='index', y='counts', size=15,
                    fill_color="grey", hover_fill_color="firebrick",
                    fill_alpha=0.1, hover_alpha=0.3,
                    line_color=None, hover_line_color="white", source=source)

p.xaxis.axis_label = 'Tag index'
p.yaxis.axis_label = 'Number of times tag appeared'

p.add_tools(HoverTool(tooltips=[("tag no.", "@index"), ("counts", "@counts")], renderers=[renderer]))

st.bokeh_chart(p)


index = list(tag_ratios.index)
ratio = list(tag_ratios.values)
source = ColumnDataSource(data=dict(index=index, ratio=ratio))

p = figure(plot_height=400,
           plot_width=600,
           title='Total % of tags to number of tags included')

p.line(x='index', y='ratio',
       line_width=1.5,
       alpha=0.7,
       color='mediumvioletred',
       source=source)

renderer = p.vbar(x='index', top='ratio',
                  width=1,
                  alpha=0.2,
                  color='mediumvioletred',
                  source=source)

p.xaxis.axis_label = 'Number of tags included'
p.yaxis.axis_label = '% of total tags'

p.add_tools(HoverTool(tooltips=[("tag no.", "@index"), ("% of total tags", "@ratio")], renderers=[renderer]))

st.bokeh_chart(p)


st.subheader("Seems like there are some rarely used tags that can be pruned when training our model, possibly set a threshold before including tags in in our analysis")


n_topics = st.slider('Number of topics', min_value=3, max_value=30, value=10, step=1)
top_tag_counts = tag_counts[:n_topics][::-1]
tags = list(top_tag_counts.index)
counts = list(top_tag_counts.values)
transparency = list(counts / max(counts))
source = ColumnDataSource(data=dict(tags=tags, counts=counts, transparency=transparency))
p = figure(y_range=tags,
           plot_height=400,
           plot_width=600,
           title=f'Top {n_topics} tags on TED')

renderer = p.hbar(y='tags',
                  right='counts',
                  height=0.8,
                  alpha='transparency',
                  color='red',
                  hover_color='lightgreen',
                  source=source)

p.ygrid.grid_line_color = None
p.x_range.start = 0

p.xaxis.axis_label = 'Number of times tag appeared'
p.yaxis.axis_label = 'Tags'

p.add_tools(HoverTool(tooltips=[("counts", "@counts")],
                      renderers=[renderer]))

st.bokeh_chart(p)


talks = list(tag_counts_per_talk.value_counts().sort_index().index)
counts = list(tag_counts_per_talk.value_counts().sort_index().values)
source = ColumnDataSource(data=dict(talks=talks, counts=counts))
p = figure(x_range=(0, max(talks)),
           plot_height=400,
           plot_width=600,
           title='Number of tags in each talk')

renderer = p.vbar(x='talks',
                  top='counts',
                  width=0.9,
                  alpha=0.8,
                  line_color='darkgrey',
                  fill_color='lightblue',
                  hover_color='orange',
                  source=source)

p.ygrid.grid_line_color = None
p.x_range.start = 0

p.xaxis.axis_label = 'Number of tags in talk'
p.yaxis.axis_label = 'Number of talks'

p.add_tools(HoverTool(tooltips=[("n_tags", "@talks"), ("counts", "@counts")],
                      renderers=[renderer]))

st.bokeh_chart(p)

st.subheader("When we do multi-label classification, we might only need to predict <20 classes")

st.subheader("Moving on to talk duration")

duration_hist, edges = np.histogram(talk_time_minutes, bins=30)
duration_data = pd.DataFrame({'n_talks': duration_hist,
                              'left': edges[:-1],
                              'right': edges[1:]})
duration_data['time_interval'] = ['%d to %d minutes' % (left, right) for left, right in
                                  zip(duration_data['left'], duration_data['right'])]
source = ColumnDataSource(data=duration_data)

p = figure(title='Histogram of talk durations in minutes',
           plot_width=600,
           x_axis_label='Duration of talk in minutes',
           y_axis_label='Number of Talks')

renderer = p.quad(bottom=0, top='n_talks',
                  left='left', right='right',
                  fill_color='violet', line_color='purple', alpha=0.6, source=source)

p.ygrid.grid_line_color = None

p.add_tools(HoverTool(tooltips=[("num talks", "@n_talks"), ("duration", "@time_interval")],
                      renderers=[renderer]))

st.bokeh_chart(p)


st.subheader("As Expected, ted talks are notorious for being strict with speaker's time, there is a sharp cutoff at 20min")

# ## Lets see if talk duration is related the number of tags

source = ColumnDataSource(data=dict(tag_counts=tag_counts_per_talk, duration=talk_time_minutes))
p = figure(plot_width=600, title='Number of tags to talk duration')

p.circle(x='duration',
         y='tag_counts',
         size=6,
         alpha=0.4,

         source=source)

p.xaxis.axis_label = "Talk duration (in minutes)"
p.yaxis.axis_label = "Number of tags"

st.bokeh_chart(p)


word_hist, edges = np.histogram(word_count['word_counts'], bins=30)
word_data = pd.DataFrame({'n_talks': word_hist,
                              'left': edges[:-1],
                              'right': edges[1:]})
word_data['word_interval'] = ['%d to %d words' % (left, right) for left, right in
                                  zip(word_data['left'], word_data['right'])]
source = ColumnDataSource(data=word_data)

p = figure(title='Histogram of word counts across transcripts',
           plot_width=600,
           x_axis_label='Number of words for transcripts',
           y_axis_label='Number of Talks')

renderer = p.quad(bottom=0, top='n_talks',
                  left='left', right='right',
                  fill_color='turquoise', line_color='blue', alpha=0.4, source=source)

p.ygrid.grid_line_color = None

p.add_tools(HoverTool(tooltips=[("num talks", "@n_talks"), ("n_words", "@left")],
                      renderers=[renderer]))

st.bokeh_chart(p)


n_speakers = st.slider('Number of speakers', min_value=3, max_value=30, value=10, step=1)
top_speaker_counts = ted_talks['speaker'].value_counts()[:n_speakers][::-1]
speakers = list(top_speaker_counts.index)
counts = list(top_speaker_counts.values)
transparency = list(counts / max(counts))
source = ColumnDataSource(data=dict(speakers=speakers, counts=counts, transparency=transparency))
p = figure(y_range=speakers,
           plot_height=400,
           plot_width=600,
           title=f'Top {n_speakers} most frequent speakers')

renderer = p.hbar(y='speakers',
                  right='counts',
                  height=0.8,
                  alpha='transparency',
                  color='orange',
                  hover_color='lightgreen',
                  source=source)

p.ygrid.grid_line_color = None
p.x_range.start = 0

p.xaxis.axis_label = 'Number of ted talks given'
p.yaxis.axis_label = 'Speakers'

p.add_tools(HoverTool(tooltips=[("counts", "@counts")],
                      renderers=[renderer]))

st.bokeh_chart(p)

st.title("Text Processing")

input_row = 22
max_char = 1000

st.subheader('Sample data')
sample_df = ted_talks[['speaker', 'duration', 'tags', 'transcript']][15:20]
st.dataframe(sample_df)

st.subheader("Sample transcript")
example_text = ted_talks['transcript'][input_row][:max_char]
st.write(example_text)

st.subheader("Cleaned transcript")
cleaned_text = ted_talks['cleaned_transcript'][input_row][:max_char]
st.write(cleaned_text)

st.subheader("Lemmatized transcript")
lemma_text = ted_talks['lemmatized_transcript'][input_row][:max_char]
st.write(lemma_text)

st.subheader("Topics extracted from Non-negative Matrix Factorization(NMF)")
st.write(["cds", "why"])

st.title("Now lets try topic modelling with our own custom input")
st.subheader("Input text (Ctrl + Enter to Submit)")
input_text = st.text_area("New transcript")

from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if input_text:
    clf = load('classifier_binary_relevance.joblib')
    body = pd.Series(input_text)
    cv = CountVectorizer().fit(body)
    article = pd.DataFrame(cv.transform(body).todense(),columns=cv.get_feature_names())
    tfidfart=TfidfTransformer().fit(article)
    art=pd.DataFrame(tfidfart.transform(article).todense())
    predictions = classifier.predict(art.astype(float))
    st.write(predictions)
    
st.subheader("Topics extracted from Non-negative Matrix Factorization(NMF)")
st.write(["cds", "why"])

st.title("Multi-label classification will be demonstrated in slides.")


