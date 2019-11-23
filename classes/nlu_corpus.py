from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)


class NLUCorpus:
    def __init__(self, texts):
        self.samples = []
        for i in texts:
            item = texts[i]
            intent = item['intent']
            slot = item['slots']
            positions = item['positions']
            text = item['text']
            tokens = tokenizer(item['text'])
            sample = NLUSample(i, text, tokens, intent, slot, positions)
            self.samples.append(sample)


class NLUToBePredict:
    def __init__(self, texts):
        self.samples = []
        for i in texts:
            item = texts[i]
            text = item['text']
            tokens = tokenizer(item['text'])
            sample = NLUSample(i, text, tokens)
            self.samples.append(sample)

    def predict(self, model):
        pass


class NLUSample:
    def __init__(self, id, text, tokens, intent=None, slots=None, positions=None):
        self.id = id
        self.text = text
        self.tokens = tokens
        self.intent = intent
        self.slots = slots
        if positions is not None:
            for item in positions:
                value = self.slots[item]
                self.slots[item] = [value, tokenizer(text[positions[item][0]:positions[item][1]])]

    def set_label(self, intent, slots, positions):
        self.intent = intent
        self.slots = slots
