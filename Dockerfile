FROM jupyter/scipy-notebook

#ADD environment.yml /tmp/environment.yml
RUN conda install --quiet --yes \
    'spacy=2.2' \
    'gensim' \
    'pyldavis' \
    'wordcloud' \
    'python-snappy' \
    'fastparquet' \
    'pip'

RUN pip install scikit-multilearn


#spaCy models, ntlk models etc
# RUN 