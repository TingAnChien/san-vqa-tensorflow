import json
import pdb

print('loading json file...')

with open('san_lstm_att.json') as data_file:
    data = json.load(data_file)

for i in xrange(len(data)):
    print i
    data[i]['question_id'] = int(data[i]['question_id'])

dd = json.dump(data,open('OpenEnded_mscoco_lstm_results.json','w'))
