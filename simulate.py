#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:20:25 2020

@author: ruud
"""


from rubbersheet import sheet

sh = sheet.Sheet((210, 297), node_spacing=40, grid_style='t', sheet_config='rubber')
ash = sheet.AnimatedLinks(sh, canvas_width_pix=400)

sh.nodes[60].impose_pos(sh.nodes[60].natural_pos - [sh.node_spacing/2, sh.node_spacing/2])
sh.nodes[15].impose_pos(sh.nodes[15].natural_pos + [sh.node_spacing/2, sh.node_spacing/2])

ash.draw()
ash.animate(timeout_iterations=1000, maxforce_limit=5)
# ash.save_as_video()
