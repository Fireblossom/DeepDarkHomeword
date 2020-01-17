from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)
import string

PUNT = set(string.punctuation)


label = open('label', 'w')
seq_in = open('seq.in', 'w')
seq_out = open('seq.out', 'w')


def BIO_gen(corpus):
    for sample in corpus.samples:
        lst = sample.tokens
        bio_lst = ['O'] * len(lst)
        for key in sample.slots:
            try:
                word = str(sample.slots[key][0])
                if word not in lst:
                    print(word)
                    if word + "'s" in lst:
                        word += "'s"
                    if word[:-1] in lst:
                        word = word[:-1]
                    if word + ',' in lst:
                        word += ','
                    if word == 'J.G.':
                        word = 'JG'
                    if word == 'U.S.':
                        word = 'US'
                    if word + '.' in lst:
                        word += '.'
                    if word == 'estaurant':
                        word = 'restaurant'
                bio_lst[lst.index(word)] = 'B-' + key
                if len(sample.slots[key]) >= 2:
                    for i in range(1, len(sample.slots[key])):
                        bio_lst[lst.index(word) + i] = 'I-' + key
            except:
                pass
        seq_out.write(' '.join(bio_lst) + '\n')
        label.write(sample.intent + '\n')
        seq_in.write(sample.text + '\n')
