import sys
import json
import copy

sample_path = sys.argv[1]

with open('../output/' + sample_path + '/json/1_first_frame_only.json', 'r') as file:
  text = file.read()
  in_dict = json.loads(text)

out_dict = copy.deepcopy(in_dict)
frame_cnt = len(in_dict['project']['vid_list'])

for i in range(1, frame_cnt + 1):
  for j, (key, value) in enumerate(in_dict['metadata'].items()):
    if i == 1:
      out_dict['metadata'].pop(key, None)
    new_key = str(i) + '_' + str(j) + key.split('_')[1]
    new_value = copy.deepcopy(value)
    new_value['vid'] = str(i)
    new_value['man'] = 1 if i == 1 else 0 
    out_dict['metadata'][new_key] = copy.deepcopy(new_value)

with open('../output/' + sample_path + '/json/2_first_frame_expanded.json', 'w') as file:
  file.write(json.dumps(out_dict))
