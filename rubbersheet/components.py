"""
Components that make up a rubber sheet model.
"""

import numpy as np
from typing import List, Iterable, Tuple, Callable


class Node:
    """Class to create a node of the sheet. The positions are in mm, the velo-
    city in mm/s, the acceleration in mm/s^2 and the mass in g, though it is 
    possible to replace mm with an arbitrary unit of length, and g with an ar-
    bitrary unit of mass.
    
    natural_pos: node's 'natural' position in the sheet as a 2-vector (x, y) in mm."""
    
    def __init__(self, natural_pos:Iterable[float]):
        self.__natural_pos = np.array(natural_pos, np.float32)
        self.__pos = np.array(natural_pos, np.float32)
        self.__vel = np.array([0., 0.], np.float32)
        self.__acc = np.array([0., 0.], np.float32)
        self.__links = []
        self.__fixed_in_place = False
        self.__drag_coefficient = 0
        self.reset_force()
    
    def set_mass(self, mass:float):
        """Set the node's mass in g. Can be approximated by the mass of the
        sheet divided by the number of nodes."""
        self.__mass = mass
    
    def set_drag(self, drag_coefficient:float):
        """Set the drag coefficient (in N/(mm/s)) of the node. Setting no drag
        on all nodes, will cause the system to indefinitely oscilate."""
        self.__drag_coefficient = drag_coefficient
    
    @property
    def links(self):
        return self.__links
    def add_link(self, link):
        self.__links.append(link)
        
    @property
    def natural_pos(self) -> np.ndarray:
        return self.__natural_pos
    
    @property
    def pos(self) -> np.ndarray:
        """Returns current position vector of the node."""
        return self.__pos
    @property
    def fixed_in_place(self) -> bool:
        """Returns True if node is fixed in place; False if it can move freely."""
        return self.__fixed_in_place
    @property 
    def displacement(self) -> np.ndarray:
        """Returns current displacement vector of node, relative to its natural position."""
        return self.__pos - self.__natural_pos
    @property
    def displacement_magnitude(self) -> float:
        """Returns magnitude of current displacement vector of node, relative to 
        its natural position. (i.e., how far away it is from its natural position."""
        return np.linalg.norm(self.__pos - self.__natural_pos)
    
    @property
    def vel(self) -> np.ndarray:
        """Returns current velocity vector of the node."""
        return self.__vel
    @property
    def acc(self) -> np.ndarray:
        """Returns current acceleration vector of the node."""
        return self.__acc
    
    def impose_pos(self, pos: Iterable[float], fix_in_place:bool=True) -> None:
        """Impose a position on the node. Fix in place indefinitely (default) or 
        let move freely."""
        self.__pos = np.array(pos, np.float32)
        self.__fixed_in_place = fix_in_place
    
    def reset_force(self) -> None:
        self.__force = np.array([0., 0.], np.float32)
        
    def add_force(self, force: Iterable[float]) -> np.ndarray:
        """Add force vector acting on node, and return currently accumulated 
        force."""
        if not self.__fixed_in_place:
            self.__force += np.array(force, np.float32)
        return self.__force
    
    @property
    def force(self) -> np.ndarray:
        """Returns current net force vector acting on node."""
        return self.__force
    @property
    def force_magnitude(self) -> float:
        """Returns magnitude of current net force vector acting on node."""
        return np.linalg.norm(self.__force)
    
    def iterate(self, dt: float = 0.1) -> float:
        """Move node based on forces acting on it, returns length of movement
        vector."""
        force = self.__force - self.__vel * self.__drag_coefficient
        self.__acc = force / self.__mass #acceleration
        self.__vel += self.acc * dt 
        self.__pos += self.vel * dt
        return np.linalg.norm(self.vel * dt)



class Link:
    """Class to create a spring-forced link between two nodes of the sheet. The
    lengths are in mm, the spring constant in N/mm, and the force in N.
    
    nodeA, nodeB: the node at each end of the link.
    """
    def __init__(self, nodeA:Node, nodeB:Node):
        self.__nodeA = nodeA
        self.__nodeB = nodeB
        self.__nodeA.add_link(self)
        self.__nodeB.add_link(self)
        self.__natural_length = np.linalg.norm(self.__nodeA.natural_pos - self.__nodeB.natural_pos)
        self.__force_magnitude = None
    
    def set_spring_constant(self, spring_constant:float) -> None:
        """Set spring constant k for linear extension/compression of the link, 
        in N/mm. It is needed to calculate the force in the link with k * Δl. It
        can calculated as follows. 
        For a 3D material, an applied force F [N] ⟂ to a surface with area A 
        [mm2] causes a relative elongation ΔL/L (in %) in the direction of the
        force. The resistance of the material to deformation is expressed with 
        Young's elasticity modulus E [N/mm2], and the deformation ΔL/L can be 
        calculated with (F/A)/E. The linear spring constant of this object can be
        calculated with k [N/mm] = F [N] / ΔL [mm] = E [N/mm2] * A [mm2] / L [mm].
        For a 2D material, an applied force F [N] ⟂ to a side with width W [mm]
        in the causes a relative extension ΔL/L (in %) of the material's length.
        Here too the elasticity modulus E can be used: ΔL/L = F/(d*W*E), with
        d [mm] being the material's thickness. So, here, the linear spring con-
        stant is k [N/mm] = F [N] / ΔL [mm] = E [N/mm2] * W [mm] * d [mm] / L 
        [mm].
        This is the spring constant for the entire sheet with width W and length
        L. We now need to change this to get the spring constant for a single 
        link. A single link represents a piece of material with length l and 
        width w, and we want to get the constant relating the expansion Δl to 
        the force f in the link. It turns out:
        k = E * w * d / l. l is the link's natural (=resting) length, which is 
        the (natural) distance between the nodes it connects. w is the width of 
        material it represents. We approximate it with sheet surface area [mm2] 
        / (number of links / 2) / link length [mm]. (The factor 2 is because
        half of the links is lengthwise, the other half width-wise.)"""
        self.__spring_constant = spring_constant
    
    @property
    def nodeA(self) -> Node:
        return self.__nodeA
    @property
    def nodeB(self) -> Node:
        return self.__nodeB
    
    @property
    def natural_length(self) -> float:
        """Length of the spring (link) at which it exerts no force."""
        return self.__natural_length
    @property 
    def length(self) -> float:
        """Current length of the link."""
        return np.linalg.norm(self.__nodeB.pos - self.__nodeA.pos)
    
    @property
    def force_magnitude(self) -> float:
        """The force in the spring as it was calculated most recently, i.e.,
        this does not recalculate the force based on current node positions.
        Directed magnitude, with >0 (<0) meaning the spring is compressed (extended)."""
        return self.__force_magnitude
    
    def __force(self) -> np.ndarray:
        """Calculates and returns force vector in spring due to current positions of nodes.
        Force vector is calculated as acting on node A. For node B, invert."""
        link_vector = self.__nodeB.pos - self.__nodeA.pos #pointing from A to B
        length = np.linalg.norm(link_vector)
        extension = self.length - self.natural_length #<0 if compressed
        force_magnitude = extension * self.__spring_constant #in [N]
        force = link_vector/length * force_magnitude #as acting on A, now as vector, still in [spaceunits/timestep]
        self.__force_magnitude = force_magnitude #Save for later
        return force
    
    def calc_and_exert_force(self, add_to_nodes:bool=True) -> None:
        """Calculate force vector in spring due to current positions of nodes,
        and add to each node (unless add_to_nodes == False)."""
        force = self.__force()
        if add_to_nodes:
            self.__nodeA.add_force(force)
            self.__nodeB.add_force(-force)
