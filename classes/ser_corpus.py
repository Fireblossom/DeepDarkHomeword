class SERCorpus:
    def __init__(self, voice):
        self.samples = []
        for item in voice:
            label = (item['valence'], item['activation'])
            features = item['features']
            sample = SERSample(features, label)
            self.samples.append(sample)


class SERToBePredict:
    def __init__(self, voice):
        self.samples = []
        for item in voice:
            features = item['features']
            sample = SERSample(features)
            self.samples.append(sample)

    def predict(self, model):
        for sample in self.samples:
            label = None
            sample.set_label(label)


class SERSample:
    def __init__(self, features, label=None):
        self.features = features
        self.label = label

    def set_label(self, label):
        self.label = label
