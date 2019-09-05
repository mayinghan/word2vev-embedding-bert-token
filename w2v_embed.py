import os
from gensim.models import Word2Vec

# create an iterable class for the file
class Sentences(object):
  def __init__(self, file_path):
    self.file_path = file_path

  def __iter__(self):
    with open(self.file_path, 'r') as f:
      for line in f:
        yield line.split()


