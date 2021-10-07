import json

folder_name = "log_folder/"

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

import os.path

def check_error_files(i):
    fname = folder_name + "error_" + str(i) + ".dat"
    if os.path.isfile(fname):
        f = open(fname)
        ln = f.read()
        f.close()
        if len(ln)>0:
            print("*********************************************")
            print(ln)
            print("*********************************************")
            print("Input number:", i)
            return True
        #if "loss is nan" in ln:
        #    print("Input number:", i)
    return False

#i = 1
file_counter = 1
n_errors = 0
for i in range(1, 6):
    '''
    input_param = dict()
    input_param["graph"] = "tree"
    input_param["graph_param_1"] = "2"
    input_param["graph_param_2"] = "4"
    '''
    
    input_param = dict()
    input_param["graph"] = "hypercube"
    input_param["graph_param_1"] = str(2+i)
    for j, metric in enumerate(activeQualityMeasures):
        if check_error_files(file_counter):
            n_errors += 1
        file_counter = file_counter + 1
    for j, metric1 in enumerate(activeQualityMeasures):
        for k, metric2 in enumerate(activeQualityMeasures):
            if j>=k:continue
            if check_error_files(file_counter):
                n_errors += 1
            file_counter = file_counter + 1

print("There are total", file_counter-1, "inputs")
print("There are total", n_errors, "errors")

