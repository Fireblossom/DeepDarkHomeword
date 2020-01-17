import io
import json

from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN

with io.open("train.json") as f:
    dataset = json.load(f)

seed = 42
engine = SnipsNLUEngine(config=CONFIG_EN, random_state=seed)
engine.fit(dataset)

dev = open('predict.json', 'w')
devdict = {}
with io.open("nlu_traindev/dev.json") as f:
    devset = json.load(f)
    for item in devset:
        t = devset[item]["text"]
        parsing = engine.parse(t)
        devdict[item] = json.loads(json.dumps(parsing, indent=2))
dev.write(json.dumps(devdict, indent=2))
