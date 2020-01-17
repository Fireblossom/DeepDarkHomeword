import json
import tarfile

from classes.nlu_corpus import NLUCorpus, NLUToBePredict
from classes.ser_corpus import SERCorpus, SERToBePredict
from collections import OrderedDict


def read_json(file_name):
    file_name = file_name.split('!')
    with tarfile.open(file_name[0]) as file:
        content = json.load(file.extractfile(file_name[1][1:]), object_pairs_hook=OrderedDict)
    if content['0'].get("text", None) is not None:  # NLU part
        if content['0'].get("intent", None) is not None:
            return NLUCorpus(content)
        else:
            return NLUToBePredict(content)
    else:  # SER part
        if content['0'].get("valence", None) is not None:
            return SERCorpus(content)
        else:
            return SERToBePredict(content)


def write_json(file_name, corpus):
    output_dict = {}
    if type(corpus) == NLUToBePredict:
        for sample in corpus.samples:
            s = {
                "intent": sample.intent,
                "text": sample.text,
                "slots": sample.slots
            }
            output_dict[sample.id] = s

    elif type(corpus) == SERToBePredict:
        for sample in corpus.samples:
            s = {
                "features": sample.features
            }
            if sample.label == 3:
                s["valence"] = 1
                s["activation"] = 1
            elif sample.label == 2:
                s["valence"] = 1
                s["activation"] = 0
            elif sample.label == 1:
                s["valence"] = 0
                s["activation"] = 1
            elif sample.label == 0:
                s["valence"] = 0
                s["activation"] = 0
            else:
                raise ValueError
            output_dict[sample.id] = s
    else:
        raise TypeError
    with open(file_name, 'w') as file:
        file.writelines(json.dumps(output_dict))


if __name__ == '__main__':
    a = read_json(
        '/Users/duan/OneDrive - Aerodefense/Uni-Stuttgart/WS19/Deep learning/DeepDarkHomeword/ser_traindev.tar.gz!/dev.json')
    gen = {}
    import random
    for i in range(len(a.samples)):
        gen[str(i)] = {"valence": random.randrange(0, 2, 1),
                       "activation": random.randrange(0, 2, 1)}
    with open('sertest.json', 'w') as f:
        import json
        f.write(json.dumps(gen, indent=2))
