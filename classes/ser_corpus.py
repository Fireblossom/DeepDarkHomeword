import numpy as np


class SERCorpus:
    def __init__(self, voice):
        self.samples = []
        for i in voice:
            item = voice[i]
            if item['valence'] == 1 and item['activation'] == 1:
                label = 3
            elif item['valence'] == 1 and item['activation'] == 0:
                label = 2
            elif item['valence'] == 0 and item['activation'] == 1:
                label = 1
            elif item['valence'] == 0 and item['activation'] == 0:
                label = 0
            else:
                raise AssertionError
            features = item['features']
            sample = SERSample(i, features, label)
            self.samples.append(sample)

    def load_data(self):
        np.random.shuffle(self.samples)
        x, y = [], []
        split_index = int(len(self.samples)/10)
        for sample in self.samples:
            x.append(sample.features)
            y.append(sample.label)
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return (x_train, y_train), (x_test, y_test)


class SERToBePredict:
    def __init__(self, voice):
        self.samples = []
        for i in voice:
            item = voice[i]
            features = item['features']
            sample = SERSample(i, features)
            self.samples.append(sample)

    def predict(self, model):
        for sample in self.samples:
            label = None
            sample.set_label(label)


class SERSample:
    def __init__(self, id, features, label=None):
        self.id = id
        self.features = features
        self.label = label

    def set_label(self, label):
        self.label = label
