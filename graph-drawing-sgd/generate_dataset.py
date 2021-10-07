import json

folder_name = "log_folder/"
map_file = "map_gd2.csv"

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

#i = 1
f = open(map_file, 'w')
file_counter = 1
for i in range(1, 6):
    '''
    input_param = dict()
    input_param["graph"] = "tree"
    input_param["graph_param_1"] = "2"
    input_param["graph_param_2"] = "4"
    '''
    
    input_param = dict()
    #input_param["graph"] = "hypercube"
    input_param["graph"] = "grid"
    #input_param["graph_param_1"] = str(2+i)
    input_param["graph_param_1"] = str(5*i)
    for j, metric in enumerate(activeQualityMeasures):
        input_param["output_file"] = folder_name + "output_" + str(file_counter) + ".txt"
        input_param["metrics"] = [metric]
    
        with open(folder_name + 'input_' + str(file_counter) + '.txt', 'w') as fp:
            json.dump(input_param, fp)

        f.write(str(file_counter)+";dev.py;"+folder_name+"input_"+str(file_counter)+".txt"+";;\n")
        file_counter = file_counter + 1
    for j, metric1 in enumerate(activeQualityMeasures):
        for k, metric2 in enumerate(activeQualityMeasures):
            if j>=k:continue
            input_param["output_file"] = folder_name + "output_" + str(file_counter) + ".txt"
            input_param["metrics"] = [metric1, metric2]
    
            with open(folder_name + 'input_' + str(file_counter) + '.txt', 'w') as fp:
                json.dump(input_param, fp)

            f.write(str(file_counter)+";dev.py;"+folder_name+"input_"+str(file_counter)+".txt"+";;\n")
            file_counter = file_counter + 1
f.close()

