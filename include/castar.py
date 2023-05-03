# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 01:38:05 2022

the normal a*
"""
# ----------------------------Astar Search Algorithm start-----------
import time
import numpy as np

class node:
    def __init__(self, number, safety, g, gh):
        self.property = (number, safety)
        self.gh = gh #real cost to come + heuristic value
        self.g = g #real cost to come
        self.path = []
    def updateC(self, gh):
        self.gh = gh


class PriorityQueue:
    def __init__(self, NodeList):
        self.queue = [] #property
        self._index = 0
        self.NodeList = NodeList
        
    def remove(self, p):
        self._index -= 1
        self.queue.remove(p)
        
    def insert(self, node):
        last = True
        for i in range(len(self.queue)):
            if self.NodeList[self.queue[i]].gh > node.gh:
                self.queue.insert(i, node.property)
                last = False
                self._index += 1
                break
        if last:
            self.queue.append(node.property)
            self._index += 1

    def pop(self):
        self._index -= 1
        return self.queue.pop(0)
    
def manhattanHeuristic(state, goal):
   """ewg, newc)
   A heuristic function estimates the cost from the current state to the nearest
   goal.  This heuristic is trivial.
   """
   return abs(goal[0] - state[0]) + abs(goal[1] - state[1]) 

def euclideanHeuristic(state, goal):
    return ((goal[0] - state[0])**2 + (goal[1] - state[1])**2)**0.5

def numberHeuristic(state, goal):
    return 1

def getCostOfActionsEuclideanDistance(a):
    return (a[0]**2 + a[1]**2)**0.5

def collisionCheck(reservedMap, position, maxDepth):
    s = 2**maxDepth
    s = s - 1
    if position[0] < 0 or position[1] < 0 or position[0] > s or position[1] > s:
        return True
    if reservedMap[position[0]][position[1]] == 99:
        return True
    return False        

def aStarSearch(xI,xG, adj_matrix, riskMap, safec = 0.9, heuristic='number'):
    "Search the node that has the lowest combined cost and heuristic first."
    """The function uses a function heuristic as an argument. We have used
  the null heuristic here first, you should redefine heuristics as part of 
  the homework. 
  Your algorithm also needs to return the total cost of the path using
  getCostofActions functions. 
  Finally, the algorithm should return the number of visited
  nodes during the search."""
  #             E
    #actions = [(1, 1), (0, 1), (-1, 1), (-1, 0), (1, 0), (1,-1), (0, -1), (-1, -1)]
    #actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    root = node(xI, 1 - riskMap[xI], 0, numberHeuristic(xI, xG))
    root.path = []
    nodeList = {} #{property：node}
    nodeList[(xI, 1 - riskMap[xI])] = root
    visited = PriorityQueue(nodeList) 
    visited.insert(root)
    count = 0
    explored = []
    while True:
        count = count + 1
        #print(count)
        if visited._index == 0:
            print("search failed")
            return False, False, False, False, False
        currentproperty = visited.pop()
        currentposition = currentproperty[0]
        currentsafety = currentproperty[1]
        if currentproperty[0] == xG:
            print("goal found")
            break
        explored.append(currentproperty)
        current = nodeList[currentproperty]
        for new, i in enumerate(adj_matrix[currentposition]):
            if i == 1:
                # check collision
                newposition = new
                newsafety = currentsafety * (1 - riskMap[newposition])
                newproperty = (newposition, newsafety)
                if newsafety > safec:
                    newg = current.g + 1
                    if heuristic == 'manhattan':
                        newc = newg + manhattanHeuristic(newposition, xG)
                    if heuristic == 'euclidean':
                        newc = newg + euclideanHeuristic(newposition, xG)
                    if heuristic == 'number':
                        newc = newg + numberHeuristic(newposition, xG)
                    # check if new node found add to nodeList and pripority queue
                    if newproperty not in nodeList:
                        newnode = node(newposition, newsafety, newg, newc)
                        visited.insert(newnode)
                        newnode.path = current.path.copy()
                        newnode.path.append(newproperty)
                        nodeList[newproperty] = newnode
                    else:
                        if newc < nodeList[newproperty].gh:
                            newnode = node(newposition, newg, newc)
                            newnode.path = current.path.copy()
                            newnode.path.append(newposition)
                            nodeList[newproperty] = newnode
                            if newproperty in  visited.queue:
                                visited.remove(newproperty)
                            visited.insert(newnode)
                        

    path = nodeList[currentproperty].path

    return path, nodeList, count, explored