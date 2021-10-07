import matplotlib.pyplot as plt
import json
import sys
import os.path
import numpy as np
import networkx as nx
import subprocess
from utils import *

map_file = "map_gd2.csv"
inputs = []
outputs = []
plot_folder = "plots/"

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

def print_uncomputed_files():
  f = open(map_file, 'r')
  l = f.readline()
  while len(l)>1:
    arr = l.split(';')
    CODE_FILE = arr[1]
    INPUT_FILE = arr[2]
    l = f.readline()

    with open(INPUT_FILE) as json_file:
      input_param = json.load(json_file)
    OUTPUT_FILE = input_param["output_file"]
    if os.path.isfile(OUTPUT_FILE):
      with open(OUTPUT_FILE) as json_file:
        output_dict = json.load(json_file)
        if "metric_value" not in output_dict.keys():
          print(INPUT_FILE)

  f.close()

#print_uncomputed_files()
#quit()

def print_metrics(metric1, metric2, sorted_keys, metric1_val, metric2_val, combined_metric1_val, combined_metric2_val):
    print("********************************")
    print(metric1)
    print("key          single          combined")
    for k in sorted_keys:
      if k in metric1_val.keys():
        single_val = metric1_val[k]
      else:
        single_val = -1
      if k in combined_metric1_val.keys():
        combined_val = combined_metric1_val[k]
      else:
        combined_val = -1
      print(k, "                ", single_val, "                ", combined_val)

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

def generate_scatter_plot(metric1, metric2, sorted_keys, metric1_val, metric2_val, combined_metric1_val, combined_metric2_val, metric1_val_metric2, metric2_val_metric1):

  #N = 50
  #x = np.random.rand(N)
  #y = np.random.rand(N)
  #colors = np.random.rand(N)
  #area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

  fig, ax = plt.subplots()
  x1, x2, x12 = [], [], []
  y1, y2, y12 = [], [], []
  colors = []
  area = []
  common_area = 20
  for k in sorted_keys:
    line_x = []
    line_y = []
    if (k in metric1_val.keys()) and (k in metric1_val_metric2.keys()):
      x1.append(metric1_val[k])
      y1.append(metric1_val_metric2[k])
      #colors.append("red")
      #area.append(common_area)
      line_x.append(x1[-1])
      line_y.append(y1[-1])

    if (k in metric2_val.keys()) and (k in metric2_val_metric1.keys()):
      x2.append(metric2_val_metric1[k])
      y2.append(metric2_val[k])
      #colors.append("blue")
      #area.append(common_area)
      line_x.append(x2[-1])
      line_y.append(y2[-1])

    if (k in combined_metric1_val.keys()) and (k in combined_metric2_val.keys()):
      x12.append(combined_metric1_val[k])
      y12.append(combined_metric2_val[k])
      #colors.append("orange")
      #area.append(common_area)
      line_x.append(x12[-1])
      line_y.append(y12[-1])

    line_ord = np.argsort(line_x)
    plt.plot([line_x[p] for p in line_ord], [line_y[p] for p in line_ord], color="black", linestyle="dashed", linewidth=1)

  #if (metric1=="stress" and metric2=="edge_uniformity") or (metric1=="stress" and metric2=="neighborhood_preservation"):
  #  print(x1, y1)

  #fig, ax = plt.subplots()
  plt.title("Hypercubes")
  plt.xlabel(metric1)
  plt.ylabel(metric2)
  sct1 = plt.scatter(x1, y1, s=common_area, c="red", alpha=0.5, marker='o')
  sct2 = plt.scatter(x2, y2, s=common_area, c="orange", alpha=0.5, marker='o')
  sct12 = plt.scatter(x12, y12, s=common_area, c="blue", alpha=0.5, marker='x')
  plt.legend((sct1, sct2, sct12),
           (metric1, metric2, 'combined'),
           scatterpoints=1,
           #loc='lower left',
           #ncol=3,
           fontsize=8)
  #plt.show()
  plt.savefig(plot_folder + metric1 + '_vs_' + metric2 + ".png")
  plt.close(fig)

TLE_files = []
graph_layout_files = set()
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
    if input_param["graph"]=="hypercube":
      G = nx.hypercube_graph(int(input_param["graph_param_1"]))

  OUTPUT_FILE = input_param["output_file"]

  if os.path.isfile(OUTPUT_FILE):
    with open(OUTPUT_FILE) as json_file:
      #metric_value = json.load(json_file)
      output_dict = json.load(json_file)
      #print(INPUT_FILE)
      if "metric_value" not in output_dict.keys():
        #print("graph_param_1", input_param["graph_param_1"], "metrics", input_param["metrics"])
        TLE_files.append(OUTPUT_FILE)
        continue
      metric_value = output_dict["metric_value"]
      metric1 = input_param["metrics"][0]
      print(input_param["metrics"], len(input_param["metrics"]))
      if len(input_param["metrics"])==2:
        metric2 = input_param["metrics"][1]
        output_dir = plot_folder + metric1 + '_vs_' + metric2 + "/"
        #output_file = plot_folder + metric1 + '_vs_' + metric2 + ".png"
      else:
        output_dir = plot_folder + metric1 + "/"
        #output_file = plot_folder + metric1 + ".png"
      if not os.path.isfile(output_dir):
        subprocess.run(["mkdir", output_dir])
      file_wo_ext = input_param["graph"] + '_' + input_param["graph_param_1"] + ".png"
      output_file = output_dir + file_wo_ext
      graph_layout_files.add(file_wo_ext)
      pos = output_dict["pos"]
      nodes = list(G.nodes())
      edge_list = [[nodes.index(u), nodes.index(v)] for u, v in G.edges()]
      draw_graph([row[0] for row in pos], [row[1] for row in pos], edge_list, output_file)

    #print(metric_value)
    inputs.append(input_param)
    outputs.append(metric_value)
  else:
    TLE_files.append(OUTPUT_FILE)

f.close()

len_meas = len(activeQualityMeasures)
for p in range(len_meas):
  for q in range(len_meas):
    if p>=q:continue
    metric1 = activeQualityMeasures[p]
    metric2 = activeQualityMeasures[q]

    '''
    plt.title("Hypercubes")
    plt.xlabel("Dimension")
    plt.ylabel(metric1 + ' (red) ' + metric2 + ' (blue)')
    plt.plot([i+3 for i in range(len(outputs))], [outputs[i]["stress"] for i in range(len(outputs))], "ro")
    '''
    print("Comparing", metric1, "and", metric2)
    metric1_val = dict()
    metric1_val_metric2 = dict()
    metric2_val = dict()
    metric2_val_metric1 = dict()
    combined_metric1_val = dict()
    combined_metric2_val = dict()
    all_keys = dict()
    for i in range(len(outputs)):
      k = int(inputs[i]["graph_param_1"])
      all_keys[k] = True
      #if (metric1=="stress" and metric2=="edge_uniformity") or (metric1=="stress" and metric2=="neighborhood_preservation"):
      #  print(inputs[i]["metrics"])
      if (metric1 in inputs[i]["metrics"]) and (metric2 in inputs[i]["metrics"]):
        #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric1]], "rs")
        #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric2]], "bs")
        combined_metric1_val[k] = outputs[i][metric1]
        combined_metric2_val[k] = outputs[i][metric2]
      elif len(inputs[i]["metrics"])==1:
        if metric1 in inputs[i]["metrics"]:
          #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric1]], "ro")
          metric1_val[k] = outputs[i][metric1]
          metric1_val_metric2[k] = outputs[i][metric2]
        elif metric2 in inputs[i]["metrics"]:
          #plt.plot([int(inputs[i]["graph_param_1"])], [outputs[i][metric2]], "bo")
          metric2_val[k] = outputs[i][metric2]
          metric2_val_metric1[k] = outputs[i][metric1]

    #plt.show()

    sorted_keys = sorted(list(all_keys.keys()))
    #print(sorted_keys)

    #print_metrics(metric1, metric2, sorted_keys, metric1_val, metric2_val, combined_metric1_val, combined_metric2_val)
    generate_scatter_plot(metric1, metric2, sorted_keys, metric1_val, metric2_val, combined_metric1_val, combined_metric2_val, metric1_val_metric2, metric2_val_metric1)
    #if (metric1=="stress" and metric2=="edge_uniformity") or (metric1=="stress" and metric2=="neighborhood_preservation"):
    #  print(metric1, metric1_val, metric1_val_metric2)
    #  print(metric2, metric2_val_metric1, metric2_val)
    #  print(combined_metric1_val, combined_metric2_val)

print(len(TLE_files), "files got TLE:", TLE_files)
print(graph_layout_files)

