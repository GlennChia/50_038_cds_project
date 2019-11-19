FROM jupyter/scipy-notebook

RUN conda install --quiet --yes \
    'spacy=2.2' \
    'gensim' \
    'pyldavis' \
    'wordcloud' \
    'python-snappy' \
    'fastparquet' \
    'inflect' \
    'pip'

RUN pip install scikit-multilearn contractions



#spaCy models, ntlk models etc
# RUN python -m spacy download en_core_web_sm
# RUN python -m spacy download en
