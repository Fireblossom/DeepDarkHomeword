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


if __name__ == '__main__':
    a = read_json('/Users/duan/OneDrive - Aerodefense/Uni-Stuttgart/WS19/Deep learning/project/nlu_traindev.tar.gz!/dev.json')
    print(type(a))
