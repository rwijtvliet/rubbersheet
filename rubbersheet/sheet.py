from rubbersheet.components import Node, Link
from typing import Iterable, Tuple, List, Union, Dict
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
import numpy as np

class Sheet:
    """Class to model a rubber sheet and simulate how it comes to rest, given
    certain imposed positions. This is done by setting up a grid of nodes, as 
    well as the links between each node and its closest neighbors. In the doc-
    strings, all units of length are stated in mm, though these can be substi-
    tuted with other units of length, or pixels. All units of mass are stated
    in g.
    
    size: (width, height) of sheet in mm.
    mass_density: mass of the sheet per unit of area, i.e. in g/mm2.
    node_spacing: how far nodes on grid should be apart, in mm.    
    grid_style: 't' (default) for triangular cells, 's' for square cells.
    sheet_config: a predefined string ('lycra' or 'rubber'), or a dict with the
        keys 'thickness' (in mm), 'mass_density' (in g/mm3) and 'young' (Young's
        elasticity modulus in N/mm2). The lower the value for young, the slower 
        the sheet will settle into a stationary state. Some values: for rubber,
        it is 100 N/mm2; for lycra, 0.5 N/mm2."""
    _sheet_configs = {'lycra': {'thickness': 0.5, 'mass_density': 1e-3, 'young': 0.5},
                      'rubber': {'thickness': 0.4, 'mass_density': 1.5e-3, 'young': 100}}
    def __init__(self, size:Iterable[float], node_spacing:float, grid_style:str='t',
                 sheet_config:Union[str,Dict]='lycra'):
        self.__size = size
        area = size[0] * size[1]
        self.__node_spacing = node_spacing
        grid_func = create_square_grid if grid_style == 's' else create_triangle_grid
        self.__nodes, self.__links = grid_func(size, node_spacing)
        #sheet_config can be Dict with material properties, or name of a preset config:
        if isinstance(sheet_config, str): 
            sheet_config = self._sheet_configs[sheet_config]
        mass_per_node = sheet_config['mass_density'] * sheet_config['thickness'] * \
            area / len(self.__nodes)
        width_per_link = area / (len(self.__links) / 2) / node_spacing #See comments at Link.set_spring_constant
        spring_constant = sheet_config['young'] * sheet_config['thickness'] * \
            width_per_link / node_spacing #node_spacing is AVERAGE link length. Maybe change for INDIVIDUAL link length.
        for node in self.__nodes:
            node.set_mass(mass_per_node)
            node.set_drag(0.3)
        for link in self.__links:
            link.set_spring_constant(spring_constant)
    
    @property
    def size(self) -> List[float]:
        """Returns the (width, height) size of the sheet in mm."""
        return self.__size
    
    @property
    def node_spacing(self) -> float:
        """Returns the spacing used to create the grid."""
        return self.__node_spacing

    @property
    def nodes(self) -> List[Node]:
        """Returns the nodes of the grid."""
        return self.__nodes
            
    @property
    def links(self) -> List[Link]:
        """Returns the links between the nodes of the grid."""
        return self.__links
    
    def do_iterations(self, num:int=1) -> Tuple[float, float]:
        """Do 'num' iterations: calculate the forces in the links, based on 
        current node positions, and update the node positions, based on the 
        forces in the links. Returns largest distance any node has moved [mm],
        and highest net force on any node [N] during the final iteration."""
        movedist_max = -np.inf
        force_max = -np.inf
        for i in range(num):
            for node in self.__nodes:
                node.reset_force()
            for link in self.__links:
                link.calc_and_exert_force()
            for node in self.__nodes:
                movedist = node.iterate(0.01)
                if i == num-1:
                    movedist_max = max(movedist_max, movedist)
                    force_max = max(force_max, node.force_magnitude)
        return movedist_max, force_max

    def interpolate():
        """Interpolate the """
        return

class AnimatedSheet:
    """Superclass to animate/draw a rubber sheet object. Not usable on its own.
    The subclasses must implement 1 argument-less methods: 
        add_frame(), which 
            calculates a list of ((x0, x1), (y0, y1), (r, g, b, a))-tuples, for 
            all lines which must be drawn, adds it to self._frames, and also 
            returns it.                
    """
    
    def __init__(self, sheet: Sheet, canvas_width_pix: int = 300):
        matplotlib.use('TkAgg') #Updating doesn't work in Qt / inline in spyder
        self._sheet = sheet
        width, height = self._sheet.size #sheet width and height in mm
        canvas_height_pix = canvas_width_pix * height / width
        
        self.__fig, self.__ax = plt.subplots()
        self.__fig.set_dpi(100)
        self.__fig.set_size_inches(canvas_width_pix / 100, canvas_height_pix / 100)
        self.__ax.set_xlim(-0.05*width, 1.05*width)#5% border in case sheet wanders.
        self.__ax.set_ylim(1.05*height, -0.05*height)
        self._frames = []
    
    def draw(self):
        """Draw current state of the sheet."""
        frame = self.add_frame() #This function must be defined in the subclass
        if len(self.__ax.get_lines()) == 0: #No lines have been added to this graph yet. Do now.
            for line in frame:
                self.__ax.add_line(plt.Line2D([], []))
        for (x, y, color), line in zip(frame, self.__ax.get_lines()):
            line.set_data(x, y)
            line.set_color(color)
        plt.pause(0.000001)
    
    def animate(self, maxforce_limit:float=0.1, timeout_iterations:int=10000, 
                timeout_sec:float=60, iterations_per_step:int=20) -> bool:
        """Animate the relaxing of the rubber sheet. Returns True if end of
        animation was reached because maximum force on all nodes dropped below
        the limit; False if a timeout was reached."""
        import datetime
        start_time = datetime.datetime.now()
        iters = 0
        while True:
            _, maxforce = self._sheet.do_iterations(iterations_per_step)
            self.draw()
            
            time = (datetime.datetime.now() - start_time).seconds
            iters += iterations_per_step
            print (f'Done {iters} iterations... max force during last iteration: {maxforce:.1f} N.') 
            #Time-out?
            if time > timeout_sec:
                print(f'Timeout due to time, after {iters} iterations and {time} seconds.')
                return False
            if iters > timeout_iterations:
                print(f'Timeout due to number of iterations, after {iters} iterations and {time} seconds.')
                return False
            #Finished?
            if maxforce < maxforce_limit:
                print(f'Finished after {iters} iterations and {time} seconds')
                return True

    def save_as_video(self, filepath:str='animation.mp4') -> None:
        """Save the animation as an mp4 video."""
        self.__ax.cla()
        print("Starting save-to-disk process (might take a while)...")
        to_draw = [[plt.plot(xs, ys, color)[0] for xs, ys, color in frame]
                   for frame in self._frames]
        ani = animation.ArtistAnimation(self.__fig, to_draw, interval=100, blit=True, repeat_delay=1000)
        ani.save(filepath, writer='ffmpeg')    
        print("Done.")



class AnimatedLinks(AnimatedSheet):
    
    __mapper = None
    
    def add_frame(self) -> Tuple:
        """Take the x and y positions and colors of all links on the graph, and 
        add to the frame-list. Returns the frame, which is a ([x0, x1], [y0, y1], 
        (r, g, b, alhpa))-tulpe."""
        if self.__mapper is None: 
            self.__calc_colormapper()
        frame = []
        for link in self._sheet.links:
            xs, ys = zip(link.nodeA.pos, link.nodeB.pos)
            color = self.__mapper.to_rgba(link.force_magnitude)
            frame.append((xs, ys, color)) #2-tuple, 2-tuple, 4-tuple
        self._frames.append(frame)
        return frame
    
    def __calc_colormapper(self):
        """Reset the color scale to reflect current force values."""
        maxabs = -np.inf
        for link in self._sheet.links:
            if link.force_magnitude is None:
                link.calc_and_exert_force(False) #Single run to calculate the force. Don't add to node!
            maxabs = max(maxabs, abs(link.force_magnitude))
        norm = matplotlib.colors.Normalize(-maxabs, maxabs, clip=True)
        self.__mapper = matplotlib.cm.ScalarMappable(norm, plt.get_cmap('coolwarm'))


class AnimatedNodes(AnimatedSheet):
    
    __mapper = None
    
    def add_frame(self) -> Tuple:
        """Take the x and y positions and colors of all links on the graph, and 
        add to the frame-list. Returns the frame, which is a ([x0, x1], [y0, y1], 
        (r, g, b, alhpa))-tulpe."""
        if self.__mapper is None: 
            self.__calc_colormapper()
        frame = []
        for node in self._sheet.nodes:
            xs, ys = zip(node.natural_pos, node.pos)
            color = self.__mapper.to_rgba(node.displacement_magnitude)
            frame.append((xs, ys, color)) #2-tuple, 2-tuple, 4-tuple
        self._frames.append(frame)
        return frame
    
    def __calc_colormapper(self):
        """Reset the color scale to reflect current displacement values."""
        maxx = -np.inf
        for node in self._sheet.nodes:
            maxx = max(maxx, abs(node.displacement_magnitude))
        norm = matplotlib.colors.Normalize(0, maxx*1.1, clip=True) #add 10% buffer to top
        self.__mapper = matplotlib.cm.ScalarMappable(norm, plt.get_cmap('cool'))



def create_square_grid(size:Iterable[float], approx_spacing:float) -> Tuple[List[Node], List[Link]]:
    """Create a list of nodes to cover sheet of certain size (width, height),
    with the nodes located on a square grid. Also create the links between them,
    with each ('inside') node connected to its closest 8 neighbours."""
    width, height = size
    cols = int(width/approx_spacing + 0.5) + 1
    col_spacing = width/(cols-1)
    rows = int(height/approx_spacing + 0.5) + 1 
    row_spacing = height/(rows-1)

    def create_and_append_link(idxA, idxB):
        nonlocal links, nodes
        nodeA = nodes[idxA]
        nodeB = nodes[idxB]
        link = Link(nodeA, nodeB, )
        links.append(link)     
    
    nodes, links = [], []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            natural_pos = (col*col_spacing, row*row_spacing)
            nodes.append(Node(natural_pos))
            if col > 0:
                create_and_append_link(idx, idx-1)      #horizontally to left
            if row > 0:
                create_and_append_link(idx, idx-cols)   #vertically up
            if row > 0 and col > 0:
                create_and_append_link(idx, idx-cols-1) #diagonally up/left
            if row > 0 and col < cols-1:
                create_and_append_link(idx, idx-cols+1) #diagonally up/right
            idx += 1
    return nodes, links



def create_triangle_grid(size: Iterable[float], approx_spacing: float) -> Tuple[List[Node], List[Link]]:
    """Create a list of nodes to cover sheet of certain size (width, height),
    with the nodes located on a triangle grid. Also create the links between them,
    with each ('inside') node connected to its closest 6 neighbours."""
    width, height = size
    cols = int(width/approx_spacing) + 2 #the odd rows will be indented by half a column
    col_spacing = width/(cols-1.5)
    rows = int(height/(0.866*approx_spacing) + 0.5) + 1 #rows must be bit nearer to keep diagonal distance same as horizontal distance.
    row_spacing = height/(rows-1)

    def create_and_append_link(idxA, idxB):
        nonlocal links, nodes
        if idxA < 0 or idxA >= len(nodes) or idxB < 0 or idxB >= len(nodes): return
        nodeA = nodes[idxA]
        nodeB = nodes[idxB]
        link = Link(nodeA, nodeB)
        links.append(link)    
        
    nodes, links = [], []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            natural_pos = np.array((col*col_spacing, row*row_spacing), np.float32)
            if (row % 2 == 0 and col == cols-1) or (row % 2 == 1 and col > 0):
                natural_pos -= (col_spacing/2, 0)
            nodes.append(Node(natural_pos))
            if col > 0:
                create_and_append_link(idx, idx-1)      #horizontally to left
            if row > 0:
                create_and_append_link(idx, idx-cols)   #up/left (up/right, straight up) on odd rows (even rows, endpoints of each row)
                if row % 2 == 0 and col < cols-1:
                    create_and_append_link(idx, idx-cols+1) #diagonally up/left on even rows
                elif row % 2 == 1 and col > 0:
                      create_and_append_link(idx, idx-cols-1) #diagonally up/right on odd rows            
            idx += 1
    # #links to cover the edges
    # for row in range(0, rows, 2):
    #     nodes.
    #     idxA = row*cols
    #     idxB = idxA + cols*2
    #     create_and_append_link(idxA, idxB)
    #     create_and_append_link(idxA-1, idxB-1)
    # #nodes/links to cover the corners
    # nodes.append(Node((width, 0))) #top right corner
    # idx = len(nodes)-1
    # create_and_append_link(idx, cols-1)
    # create_and_append_link(idx, 2*cols-1)
    # if rows % 2 == 0: #even number of rows, empty corner in bottom left
    #     nodes.append(Node((0, height))) #top right corner
    #     create_and_append_link(idx+1, (rows-1)*cols)
    #     create_and_append_link(idx+1, (rows-2)*cols)
    # else: #odd number of rows, empty corner in bottom right
    #     nodes.append(Node((width, height)))
    #     create_and_append_link(idx+2, rows*cols-1)
    #     create_and_append_link(idx+2, (rows-1)*cols-1)
    
    
    return nodes, links