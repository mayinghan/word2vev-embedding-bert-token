import os
import sys
import string
import re

from pytorch_transformers import BertTokenizer

def convert_corpus_to_tokens(filepath):
  with open('tokened_mimic.txt', 'w') as w:
    with open(filepath, 'r') as f:
      for line in f:
        token_list = tokenize(line)
        new_token_sentence = ' '.join(token_list) + '\n'
        w.write(new_token_sentence)
      
        
    
def tokenize(word) -> list:
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased'
                , do_lower_case=False)

  return tokenizer.tokenize(word)


if __name__ == "__main__":
    convert_corpus_to_tokens(os.path.join('./mimiciii_notes_new.sent.lowered.all0.txt'))