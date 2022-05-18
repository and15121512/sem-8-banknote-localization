import sys
import json

sample_path = sys.argv[1]

with open('../output/' + sample_path + '/json/4_ground_truth.json', 'r') as file:
  text = file.read()
  json_text = json.loads(text)
  gt_dict = json_text['metadata']
  frame_cnt = len(json_text['project']['vid_list'])

with open('../output/' + sample_path + '/json/3_tracker_output.json', 'r') as file:
  text = file.read()
  json_text = json.loads(text)
  tr_dict = json_text['metadata']
  width = int(json_text['project']['resolution'][0])
  height = int(json_text['project']['resolution'][1])

dist_total = 0
dists = [ 0 ] * frame_cnt
mans = [ 0 ] * frame_cnt
for key, gt_value in gt_dict.items():
  frame_idx_gt = int(gt_value['vid']) - 1
  tr_value = tr_dict[key]
  mans[frame_idx_gt] = tr_value['man']
  for i in range(1, 9, 2):
    x_gt = gt_value['xy'][i]
    y_gt = gt_value['xy'][i + 1]
    x_tr = tr_value['xy'][i]
    y_tr = tr_value['xy'][i + 1]
    dists[frame_idx_gt] += (abs(x_gt - x_tr) + abs(y_gt - y_tr)) / float(width)
    dist_total += (abs(x_gt - x_tr) + abs(y_gt - y_tr)) / float(width)

with open('../output/' + sample_path + '/metrics.csv', 'w') as file:
  for i, (dist, man) in enumerate(zip(dists, mans)):
    file.write(sample_path + '/frames/frame' + str(i) + '.png,' + str(dist) + ',' + str(man) + '\n')

print('Total metrics:', dist_total)
