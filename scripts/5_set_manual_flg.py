import sys
import json

sample_path = sys.argv[1]
frame_num = sys.argv[2]
new_flg_val = '' if len(sys.argv) < 4 else sys.argv[3]

with open('../output/' + sample_path + '/json/2_first_frame_expanded.json', 'r') as file:
  text = file.read()
  in_dict = json.loads(text)

for key, value in in_dict['metadata'].items():
  if str(int(frame_num) + 1) != value['vid']:
    continue
  value['man'] = 0 if new_flg_val == 0 else 1
  break

with open('../output/' + sample_path + '/json/2_first_frame_expanded.json', 'w') as file:
  file.write(json.dumps(in_dict))
