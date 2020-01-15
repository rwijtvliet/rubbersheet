#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:02:10 2020

@author: ruud
"""


import sheet as rs

sh = rs.Sheet((210, 297), node_spacing=40, grid_style='t', sheet_config='rubber')
ash = rs.AnimatedSheet_Links(sh, canvas_width_pix=400)

sh.nodes[32].impose_pos(sh.nodes[32].natural_pos + [20, 30])
sh.nodes[16].impose_pos(sh.nodes[16].natural_pos - [sh.node_spacing/2, sh.node_spacing/2])

# ash.draw()
ash.animate(timeout_iterations=100)
ash.draw()

#%%
import matplotlib as mpl
from datetime import datetime
import numpy as np

def weights(x1, y1, x2, y2, x3, y3, x, y):
    """Returns weights for interpolation in triangle."""
    if x == x1 and y == y1: return 1, 0, 0
    if x == x2 and y == y2: return 0, 1, 0
    if x == x3 and y == y3: return 0, 0, 1
    d = (y2-y3)*(x1-x3) + (x3-x2)*(y1-y3) #denominator
    if abs(d) < 0.001: d = 0.001 #also keep very small negative values
    w1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / d
    w2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / d
    if w1 < -0.01 or w2 < -0.01 or w1+w2>1.01:
        return w1, w2, 1-w1-w2
    # else:
    #     d1 = 1/np.sqrt((x-x1)**2 + (y-y1)**2)
    #     d2 = 1/np.sqrt((x-x2)**2 + (y-y2)**2)
    #     d3 = 1/np.sqrt((x-x3)**2 + (y-y3)**2)
    #     s = sum([d1, d2, d3])        
    #     return d1/s, d2/s, d3/s
    return w1, w2, 1-w1-w2

def interpolation_function(node1, node2, node3, pix_per_mm):
    """Returns a bounding box (xrange and yrange) for the 3 supplied nodes, as
    well as a function which takes an x and a y as input, and returns the 
    interpolated value at that location, based on the values of 3 nodes."""
    vals = np.array([node.displacement * pix_per_mm for node in [node1, node2, node3]])
    x1, y1 = node1.natural_pos * pix_per_mm
    x2, y2 = node2.natural_pos * pix_per_mm
    x3, y3 = node3.natural_pos * pix_per_mm
    x_range = range(int(min(x1, x2, x3)), int(max(x1, x2, x3)))
    y_range = range(int(min(y1, y2, y3)), int(max(y1, y2, y3)))
    
    def interpolate(x: int, y: int):
        weight = weights(x1, y1, x2, y2, x3, y3, x, y)
        if sum([w < -0.01 for w in weight]) > 0: 
            return np.nan #np.vectorize turns None into np.nan anyway...
        else:
            return sum(w*v for w, v in zip(weight, vals))
    
    return x_range, y_range, interpolate


np.set_printoptions(precision=0)
pix_per_mm = 2
size_pixels = (np.array(sh.size) * pix_per_mm).astype(int)
holder = np.array([0., 0.], np.float32)
pixels = np.full((*size_pixels, 2), np.nan, np.float32)

start_time = datetime.now()
nodecombinations = set()

for node1 in sh.nodes:
    for link1 in node1.links:
        node2 = link1.nodeB if node1 == link1.nodeA else link1.nodeA
        for link2 in node2.links:
            node3 = link2.nodeB if node2 == link2.nodeA else link2.nodeA
            for link3 in node3.links:
                node4 = link3.nodeB if node3 == link3.nodeA else link3.nodeA
                if node4 == node1:
                    #Found a loop of 3 nodes that make a triangle
                    combo = frozenset({node1, node2, node3})
                    if combo in nodecombinations:
                        continue
                    nodecombinations.add(combo)
                    
                    xrange, yrange, itp = interpolation_function(node1, node2, node3, pix_per_mm)
                    
                    # itp = np.vectorize(itp) #Vectorize: ~3x faster than looping
                    # xx = np.array([xrange for y in yrange], dtype=int).T
                    # yy = np.array([yrange for x in xrange], dtype=int)
                    # # xx = np.array([[xrange for y in yrange] for i in range(2)], dtype=int).T
                    # # yy = np.array([np.array([yrange for x in xrange], dtype=int).T for i in range(2)]).T
                    # result = np.ndarray(pixels.shape)
                    # result[xx, yy] = itp(xx, yy)
                    # pixels[xx, yy] = np.where(np.isnan(result), pixels[xx, yy], result)
                    for x in xrange:
                        for y in yrange:
                            result = itp(x, y)
                            if not np.isnan(result).any():
                                pixels[x, y] = result
print(f'that took {(datetime.now()-start_time).seconds} seconds.')
#%%

im = np.where(np.isnan(pixels), np.inf, pixels)
im = im.astype(np.float32) 
im = np.apply_along_axis(np.linalg.norm, 2, im) #2D
im = np.where(im == np.inf, im.min(), im)
im = np.interp(im, (im.min(), im.max()), (0, +1))
mpl.image.imsave('interpolated_inversedistance.png', im)

                
    