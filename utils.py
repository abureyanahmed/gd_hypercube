# some utility function

from edge_crossing import *
from input_functions import *
import matplotlib.pyplot as plt

def number_of_intersections(file_name):
 #print(file_name)
 if file_name.endswith('.dot'):
  node_coords, edge_list = parse_dot_file(file_name)
 elif file_name.endswith('.txt'):
  n, node_coords, edge_list = take_input(file_name)
 count = 0
 for i in range(len(edge_list)):
  for j in range(i+1,len(edge_list)):
   edge1 = edge_list[i]
   edge2 = edge_list[j]
   #if not share_vertex(edge1, edge2):
   if(doIntersect(node_coords[edge1[0]][0], node_coords[edge1[0]][1], node_coords[edge1[1]][0], node_coords[edge1[1]][1], node_coords[edge2[0]][0], node_coords[edge2[0]][1], node_coords[edge2[1]][0], node_coords[edge2[1]][1])):
    count = count + 1
    #print(str(edge1)+" intersects "+str(edge2))
 return count


def make_segment(p1x, p1y, p2x, p2y):
 segment = (p1x, p1y)
 segment = (segment,) + ((p2x, p2y),)
 return segment

def min_angle(file_name):
 #print(file_name)
 if file_name.endswith('.dot'):
  node_coords, edge_list = parse_dot_file(file_name)
 elif file_name.endswith('.txt'):
  n, node_coords, edge_list = take_input(file_name)
 elif file_name.endswith('.json'):
  n, node_coords, edge_list = take_input_from_json(file_name)
 min_ang = 180
 segment_arr = []
 for i in range(len(edge_list)):
  edge1 = edge_list[i]
  segment_arr.append(make_segment(node_coords[edge1[0]][0], node_coords[edge1[0]][1], node_coords[edge1[1]][0], node_coords[edge1[1]][1]))
 #print(segment_arr)
 isect = poly_point_isect.isect_segments_include_segments(segment_arr)
 for x in isect:
  intersection, segment_pair = x
  #seg1, seg2 = segment_pair
  for i in range(len(segment_pair)-1):
   for j in range(i+1, len(segment_pair)):
    seg1 = segment_pair[i]
    seg2 = segment_pair[j]
    p1, q1 = seg1
    p1x, p1y = p1
    q1x, q1y = q1
    p2, q2 = seg2
    p2x, p2y = p2
    q2x, q2y = q2
    if(doIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)):
     r = getIntersection(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)
     if r == False:continue
     angle = getAngleLineSeg(p1x, p1y, p2x, p2y, r[0], r[1])
     supplementary_angle = 180 - angle
     angle = min(angle, supplementary_angle)
     if angle == -1:
      return -1
     if min_ang>angle:
      min_ang = angle
 #print(isect)
 return min_ang


my_inf = 10000000

def min_angle_simple(file_name):
 if file_name.endswith('.txt'):
  n, coord_list, edge_list = take_input(file_name)
 elif file_name.endswith('.json'):
  n, coord_list, edge_list = take_input_from_json(file_name)
 elif file_name.endswith('.dot'):
  n, coord_list, edge_list = take_input_from_dot(file_name)
 min_ang = 180
 for i in range(len(edge_list)):
  for j in range(i+1,len(edge_list)):
   p1x = coord_list[edge_list[i][0]][0]
   p1y = coord_list[edge_list[i][0]][1]
   q1x = coord_list[edge_list[i][1]][0]
   q1y = coord_list[edge_list[i][1]][1]
   p2x = coord_list[edge_list[j][0]][0]
   p2y = coord_list[edge_list[j][0]][1]
   q2x = coord_list[edge_list[j][1]][0]
   q2y = coord_list[edge_list[j][1]][1]
   if doIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    #print(''.join(str(e)+',' for e in edge_list[i]) + ' intersects ' + ''.join(str(e)+',' for e in edge_list[j]))
    #print('p1x:'+str(coord_list[edge_list[i][0]][0])+',p1y:'+str(coord_list[edge_list[i][0]][1])+',q1x:'+str(coord_list[edge_list[i][1]][0])+',q1y:'+str(coord_list[edge_list[i][1]][1])+',p2x:'+str(coord_list[edge_list[j][0]][0])+',p2y:'+str(coord_list[edge_list[j][0]][1])+',q2x:'+str(coord_list[edge_list[j][1]][0])+',q2y'+str(coord_list[edge_list[j][1]][1]))
    r = getIntersection(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)
    if r == False:continue
    angle = getAngleLineSeg(p1x, p1y, p2x, p2y, r[0], r[1])
    supplementary_angle = 180 - angle
    angle = min(angle, supplementary_angle)
    if angle == -1:
     return -1
    if min_ang>angle:
     min_ang = angle
 return min_ang


def draw_graph(x, y, edge_list, output_file):
 global my_inf
 fig, ax = plt.subplots()
 plt.axis('off')
 xmin = my_inf
 xmax = -1
 ymin = my_inf
 ymax = -1
 for i in range(len(x)):
  if xmin > x[i]: xmin = x[i]
  if xmax < x[i]: xmax = x[i]
  if ymin > y[i]: ymin = y[i]
  if ymax < y[i]: ymax = y[i]
 for i in range(len(x)):
  circle = plt.Circle((x[i], y[i]), (xmax+ymax-xmin-ymin)/200.00, clip_on=False)
  ax.add_artist(circle)
  #ax.text(x[i]+1, y[i]+1, str(i), fontsize=12)
 for i in range(len(edge_list)):
  ax.plot([x[edge_list[i][0]], x[edge_list[i][1]]], [y[edge_list[i][0]], y[edge_list[i][1]]], 'k-', lw=2)
 ax.set_xlim([min(xmin,ymin),max(xmax,ymax)])
 ax.set_ylim([min(xmin,ymin),max(xmax,ymax)])
 fig.savefig(output_file)
 plt.close(fig)

