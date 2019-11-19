class NLUCorpus:
    def __init__(self, texts):
        self.samples = []
        for item in texts:
            intent = item['intent']
            slot = item['slots']
            positions = item['positions']
            text = item['text']
            sample = NLUSample(text, intent, slot, positions)
            self.samples.append(sample)


class NLUToBePredict:
    def __init__(self, texts):
        self.samples = []
        for item in texts:
            text = item['text']
            sample = NLUSample(text)
            self.samples.append(sample)


class NLUSample:
    def __init__(self, text, intent=None, slots=None, positions=None):
        self.text = text
        self.intent = intent
        self.slots = slots
        self.positions = positions

    def set_label(self, intent, slots, positions):
        self.intent = intent
        self.slots = slots
        self.positions = positions
