import util.IO_util
from util.BIO_gen import BIO_gen

corpus = util.IO_util.read_json(
    '/Users/duan/OneDrive - Aerodefense/Uni-Stuttgart/WS19/Deep learning/DeepDarkHomeword/nlu_traindev.tar.gz!/./nlu_traindev/train.json')
BIO_gen(corpus)
