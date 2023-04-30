import json

o = json.load(open("arguments.json"))
r = json.load(open("resolved_arguments.json"))

for i in range(len(o['lines'])):
    original = o['lines'][i]
    arguments = []

    for arg in original['arguments']:
        arguments.append(arg['text'])
    
    if len(arguments) == 0:
        r['lines'][i]['text'] = o['lines'][i]['text']


d = json.dumps(r, indent=2)
with open("r.json", "w") as file:
    file.write(d)