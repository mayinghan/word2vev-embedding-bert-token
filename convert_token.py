import os
import sys

from pytorch_transformers import BertTokenizer

def read_corpus():
  pass

def tokenize():
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased'
                , do_lower_case=False)

  print(tokenizer.tokenize('i love u'))

if __name__ == "__main__":
    tokenize()