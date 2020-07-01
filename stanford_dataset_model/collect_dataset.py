import requests
import tarfile
import os
r = requests.get('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
with open('aclImdb_v1.tar.gz', 'wb') as f:
    f.write(r.content)
z = tarfile.open('aclImdb_v1.tar.gz', 'r:gz')
os.mkdir('imdb_movie_review')
z.extractall('imdb_movie_review')