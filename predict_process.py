import io
import json

with io.open("predict.json") as f:
    dataset = json.load(f)

to_be_upload = {}

for item in dataset:
    predict = {
        #"text": dataset[item]["input"],
        "intent": dataset[item]["intent"]["intentName"],
        "slots": {}
    }
    for slot in dataset[item]["slots"]:
        predict["slots"][slot["slotName"]] = slot["rawValue"]
    to_be_upload[item] = predict

with open("upload.json", 'w') as file:
    file.write(json.dumps(to_be_upload, indent=2))
