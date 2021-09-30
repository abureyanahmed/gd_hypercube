import matplotlib.pyplot as plt
import json
import sys
import os.path
map_file = "map_gd2.csv"
inputs = []
outputs = []

activeQualityMeasures = [
    'stress',
    'edge_uniformity',
    'neighborhood_preservation',
    'crossings',

    'crossing_angle_maximization',
    'aspect_ratio',
    'angular_resolution',
    'vertex_resolution',
    'gabriel',

]

len_meas = len(activeQualityMeasures)
for p in range(len_meas):
  for q in range(len_meas):
    if p>=q:continue
    metric1 = activeQualityMeasures[p]
    metric2 = activeQualityMeasures[q]

    f = open(map_file, 'r')
    l = f.readline()
    while len(l)>1:
      arr = l.split(';')
      CODE_FILE = arr[1]
      INPUT_FILE = arr[2]
      FILE_NAME = arr[3]
      OUTPUT_FILE = arr[4]
      l = f.readline()

      # Opening JSON file
      with open(INPUT_FILE) as json_file:
        input_param = json.load(json_file)

      OUTPUT_FILE = input_param["output_file"]

      if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE) as json_file:
          metric_value = json.load(json_file)

        #print(metric_value)
        inputs.append(input_param)
        outputs.append(metric_value)

    '''
    plt.title("Hypercubes")
    plt.xlabel("Dimension")
    plt.ylabel(metric1 + ' (red) ' + metric2 + ' (blue)')
    plt.plot([i+3 for i in range(len(outputs))], [outputs[i]["stress"] for i in range(len(outputs))], "ro")
    '''
    print("Comparing", metric1, "and", metric2)
    metric1_val = dict()
    metric2_val = dict()
    combined_metric1_val = dict()
    combined_metric2_val = dict()
    all_keys = dict()
    for i in range(len(outputs)):
      k = int(inputs[i]["graph_param_1"])
      all_keys[k] = True
      if (metric1 in inputs[i]["metrics"]) and (metric2 in inputs[i]["metrics"]):
        #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric1]], "rs")
        #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric2]], "bs")
        combined_metric1_val[k] = outputs[i][metric1]
        combined_metric2_val[k] = outputs[i][metric2]
      elif len(inputs[i]["metrics"])==1:
        if metric1 in inputs[i]["metrics"]:
          #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric1]], "ro")
          print(i, metric1, outputs[i])
          metric1_val[k] = outputs[i][metric1]
        elif metric2 in inputs[i]["metrics"]:
          #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric2]], "bo")
          metric2_val[k] = outputs[i][metric2]

    #plt.show()

    sorted_keys = sorted(list(all_keys.keys()))
    #print(sorted_keys)

    print("********************************")
    print(metric1)
    print("key		single		combined")
    for k in sorted_keys:
      if k in metric1_val.keys():
        single_val = metric1_val[k]
      else:
        single_val = -1
      if k in combined_metric1_val.keys():
        combined_val = combined_metric1_val[k]
      else:
        combined_val = -1
      print(k, "		", single_val, "		", combined_val)

    print("********************************")
    print(metric2)
    print("key              single          combined")
    for k in sorted_keys:
      if k in metric2_val.keys():
        single_val = metric2_val[k]
      else:
        single_val = -1
      if k in combined_metric2_val.keys():
        combined_val = combined_metric2_val[k]
      else:
        combined_val = -1
      print(k, "            ", single_val, "                ", combined_val)


