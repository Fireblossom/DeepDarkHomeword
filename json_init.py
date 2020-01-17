import json
from util.IO_util import read_json

json_str = {
                "entities": {},
                "language": "en",
                "intents": {
                    'RateBook': {"utterances": []},
                    'SearchScreeningEvent': {"utterances": []},
                    'AddToPlaylist': {"utterances": []},
                    'BookRestaurant': {"utterances": []},
                    'SearchCreativeWork': {"utterances": []},
                    'PlayMusic': {"utterances": []},
                    'GetWeather': {"utterances": []}
                }
            }

corpus = read_json('/Users/duan/OneDrive - Aerodefense/Uni-Stuttgart/WS19/Deep learning/DeepDarkHomeword/nlu_traindev.tar.gz!/./nlu_traindev/train.json')
entity = {}
for sample in corpus.samples:
    spl = []
    for item in sample.positions:
        spl.append(sample.positions[item])

    text = []
    f = 0
    for i in spl:
        if f == 0 and i[0] != 0:
            text.append(sample.text[0:i[0]])
        elif f < i[0]:
            text.append(sample.text[f:i[0]])
        text.append(sample.text[i[0]:i[1]+1])
        f = i[1]+1
    if f < len(sample.text):
        text.append(sample.text[f:])

    data = []
    for t in text:
        if t not in sample.slots.values():
            data.append({"text": t})
        else:
            for item in sample.slots:
                if sample.slots[item] == t:
                    data.append({"text": t, "entity": item, "slot_name": item})

    data = {'data': data}
    json_str["intents"][sample.intent]["utterances"].append(data)
    for item in sample.slots:
        if item not in entity:
            entity[item] = [sample.slots[item]]
        elif sample.slots[item] not in entity[item]:
            entity[item].append(sample.slots[item])

for e in entity:
    json_str["entities"][e] = {"automatically_extensible": True,
                               "use_synonyms": True,
                               "matching_strictness": 1.0,
                               "data": []}
    for s in entity[e]:
        json_str["entities"][e]["data"].append({"value": s,
                                                "synonyms": []})

with open('train.json', 'w') as file:
    s = json.dumps(json_str, indent=2)
    file.write(s)
