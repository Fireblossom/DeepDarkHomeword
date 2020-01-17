from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)
PUNT = set(['.', '?', '!'])


class NLUCorpus:
    def __init__(self, texts):
        self.samples = []
        for i in texts:
            item = texts[i]
            intent = item['intent']
            slots = item['slots']
            positions = item['positions']
            text = item['text']
            text1 = ''.join(ch for ch in text if ch not in PUNT)
            tokens = tokenizer(text1)
            sample = NLUSample(i, text, tokens, intent, slots, positions)
            self.samples.append(sample)


class NLUToBePredict:
    def __init__(self, texts):
        self.samples = []
        for i in texts:
            item = texts[i]
            text = item['text']
            text1 = ''.join(ch for ch in text if ch not in PUNT)
            tokens = tokenizer(text1)
            sample = NLUSample(i, text, tokens)
            self.samples.append(sample)

    def predict(self, model):
        pass


class NLUSample:
    def __init__(self, id, text, tokens, intent=None, slots=None, positions=None):
        self.id = id
        self.text = text
        self.tokens = list([str(t) for t in tokens])
        self.intent = intent
        self.slots = slots
        self.positions = positions
        if positions is not None:
            for item in positions:
                lst = list(tokenizer(text[positions[item][0]:positions[item][1]+1]))
                # self.slots[item] = list([str(t) for t in lst])
                self.slots[item] = text[positions[item][0]:positions[item][1]+1] #  [value, tokenizer(text[positions[item][0]:positions[item][1]+1])]

    def set_label(self, intent, slots):
        self.intent = intent
        self.slots = slots
