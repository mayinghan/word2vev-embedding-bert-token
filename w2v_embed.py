import os
from gensim.models import Word2Vec
from pytorch_transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-cased'
                  , do_lower_case=False)

# create an iterable class for the file
class Sentences(object):
  def __init__(self, content):
    self.file_path = content

  def __iter__(self):
    with open(self.file_path, 'r') as f:
      for line in tqdm(f, desc='line'):
        yield self.tokenize(line)

  def tokenize(self, word) -> list:
    return tokenizer.tokenize(word)


def main():
  sentences = Sentences('./mimiciii_notes_new.sent.lowered.all0.txt')
  model = Word2Vec(sentences, size=50, min_count=5, workers=3, sg=1, iter=3)
  model.save('./model.model')

  # write embedding to output file
  with open('mimic_emb.txt', 'w') as w:
    for word in model.wv.index2word:
      emb_list = [word]
      emb_list.extend(model.wv[word].tolist())
      w.write(' '.join(str(v) for v in emb_list))
      w.write('\n')

if __name__ == "__main__":
  main()