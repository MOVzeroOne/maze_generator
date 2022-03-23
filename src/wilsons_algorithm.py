import torch 
import numpy as np 
from collections import defaultdict
import matplotlib.pyplot as plt 

class maze_generator():
    """
    implementation of Wilson's algorithm for maze generation    
    """
    def __init__(self,num_width_cells,num_height_cells):
        self.num_width_cells = num_width_cells
        self.num_height_cells = num_height_cells
        self.width = self.num_width_cells*2+1
        self.height = self.num_height_cells*2 +1
    
    def _set_cell_road(self,node):
        """sets a cell to be a road"""
        self.maze[node[1]][node[0]] = 1
    
    def _set_cell_wall(self,node):
        """sets a cell to be a wall"""
        self.maze[node[1]][node[0]] = 0

    def _sample_univisted_node(self):
        """returns a randomly sampled unvisted node"""
        return self.unvisted_list[np.random.choice(np.arange(len(self.unvisted_list)))]

    def _visit(self,node):
        """sets a node as visited and sets it as a road (as walls cant be visited)"""
        if(not self.visited[node]):
            self.unvisted_list.remove(node)
            self.visited[node] = True
            self._set_cell_road(node)

    def generate(self):
        self.maze = torch.zeros(self.width,self.height)
        self.visited = defaultdict(bool)
        self.unvisted_list = [(x,y) for x in range(self.width) for y in range(self.height) if (x % 2 == 1 and y % 2 == 1)]
        self.directions = defaultdict(tuple)

        self.start_node = self._sample_univisted_node()
        self._visit(self.start_node)

        plt.ion()
        while(len(self.unvisted_list) > 0):
            self._random_walk()
            plt.cla()
            plt.imshow(self.maze)
            plt.pause(0.1)
        return self.maze 
        
    def _valid_node(self,node):
        """tests if a node is valid/legal"""
        return node[0] >= 1 and node[1] >= 1 and node[0] < self.width and node[1] < self.height

    def _generate_neighbors(self,node):
        """returns a list of neighbors"""
        up_neighbor = (node[0],node[1]+2)
        down_neighbor = (node[0],node[1]-2)
        left_neighbor = (node[0]-2,node[1])
        right_neighbor = (node[0]+2,node[1])
        neighbor_list = [neighbor for neighbor in [up_neighbor,down_neighbor,left_neighbor,right_neighbor] if(self._valid_node(neighbor))]
        return neighbor_list
    
    def _random_walk(self):
        start_node = self._sample_univisted_node()
        current_node = start_node
        neighbor_list = self._generate_neighbors(current_node)

        while(True):
            chosen_neighbor = neighbor_list[np.random.choice(np.arange(len(neighbor_list)))]
            self.directions[current_node] = chosen_neighbor

            if(self.visited[chosen_neighbor]):
                break
            else:
                current_node = chosen_neighbor
                neighbor_list = self._generate_neighbors(current_node)
        self.create_path_from_directions(start_node,current_node)

    def create_path_from_directions(self,start_node,end_node):
        """makes a path following the directions from start node till the end node"""
        current_node = start_node
        while(True):
            if(not (current_node == end_node)):
                next_node = self.directions[current_node]
                self._connect_adjacent_nodes(current_node, next_node)
                current_node = next_node
            else:
                next_node = self.directions[current_node]
                self._connect_adjacent_nodes(current_node, next_node)
                break
        

    def _connect_adjacent_nodes(self,node1,node2):
        """connect two adjacent nodes with a road"""
        if(node1[0] > node2[0] and node1[1] == node2[1]):
            #node1 on the right
            x = node1[0] - 1
            y = node1[1]

        elif(node1[0] < node2[0] and node1[1] == node2[1]):
            #node1 on the left
            x = node1[0] + 1
            y = node1[1]
        elif(node1[1] > node2[1] and node1[0] == node2[0]):
            #node1 above
            x = node1[0]
            y = node1[1] - 1 
        elif(node1[1] < node2[1] and node1[0] == node2[0]):
            #node1 below
            x = node1[0]
            y = node1[1] + 1
        else:
            return
        self._visit(node1)
        self._visit(node2)
        self._set_cell_road((x,y))    

def floodfill(maze):
    non_zero_pos_list = (maze.T == 1).nonzero()
    default_dict_maze = defaultdict(bool)
    visted_pos = defaultdict(bool)
    for pos in non_zero_pos_list:
        default_dict_maze[(pos[0].item(),pos[1].item())] = True
    
    direction_list = [torch.tensor([1.0,0.0]).type(torch.long),torch.tensor([-1.0,0.0]).type(torch.long),torch.tensor([0.0,1.0]).type(torch.long),torch.tensor([0.0,-1.0]).type(torch.long)]
    start_pos = non_zero_pos_list[np.random.choice(np.arange(len(non_zero_pos_list)))]
    frontier = [start_pos]
    

    maze_overlay = torch.zeros([*maze.size(),1])
    distance = 0
    plt.ion()
    while(len(frontier) > 0):
        for pos in frontier:
            maze_overlay[pos[1],pos[0]] += distance 
            visted_pos[(pos[0].item(),pos[1].item())] = True

        frontier = [(pos + direction) for pos in frontier for direction in direction_list if default_dict_maze[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())] and not visted_pos[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())]]
        distance +=1 
        plt.cla()
        plt.imshow(maze_overlay)
        plt.pause(0.01)
    
    plt.savefig("analysis.png",bbox_inches='tight')


if __name__ == "__main__":
    gen = maze_generator(100, 100)
    floodfill(gen.generate())