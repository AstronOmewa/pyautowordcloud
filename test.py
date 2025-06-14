import json

d = json.decoder.JSONDecoder()

params = d.decode(open('param.json').read())
print(params['dpi'])