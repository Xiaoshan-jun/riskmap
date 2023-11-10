# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 01:38:05 2022

the normal a* with safety constrain
"""
# ----------------------------Astar Search Algorithm start-----------
import time
import numpy as np

class node:
    def __init__(self, position, g, gh):
        self.position = position
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
            if self.NodeList[(self.queue[i][0], self.queue[i][1])][self.queue[i][2]].gh > node.gh:
                self.queue.insert(i, node.position)
                last = False
                self._index += 1
                break
        if last:
            self.queue.append(node.position)
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

def learnedHeuristic(des, hmap, size):
    # if des[2] >= 1:
    #     k = 0
    # elif des[2] >= 0.98:
    #     k = 1
    # elif des[2] >= 0.96:
    #     k = 2
    # elif des[2] >= 0.94:
    #     k = 3
    # elif des[2] >= 0.92:
    #     k = 4
    # else:
    #     return 100000
    #print(state)
    #print(hmap[state])
    if hmap[des[0]][des[1]] >= 0:
        #print(des)
        #print(hmap[des[0]][des[1]][k])
        return hmap[des[0]][des[1]]
    else:
        return 100000

def getCostOfActionsEuclideanDistance(a):
    return (a[0]**2 + a[1]**2)**0.5

def collisionCheck(reservedMap, position):
    s = len(reservedMap)
    m = len(reservedMap[0])
    s = s - 1
    if position[0] < 0 or position[1] < 0 or position[0] > s or position[1] > m - 1:
        return True
    return False        

def aStarSearch(xI,xG, riskMap, safec = 0.9,heuristic='manhattan', hmap = None):
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
    actions = [ ( 0, 1), ( -1, 0), ( 1, 0),  ( 0, -1)]
    #actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    root = node(xI, 0, manhattanHeuristic(xI, xG))
    root.path = []
    nodeList = {} #{position：{risk: node} }
    nodeList[(xI[0], xI[1])] = {}
    nodeList[(xI[0], xI[1])][xI[2]] = root
    visited = PriorityQueue(nodeList) 
    visited.insert(root)
    count = 0
    explored = []
    while True:
        count = count + 1
        #print(count)
        if visited._index == 0:
            #print("search failed")
            return False, False, False, False, False
        currentposition = visited.pop()
        if currentposition[0] == xG[0] and currentposition[1] == xG[1]:
            xG = currentposition
            #print("goal found")
            break
        explored.append(currentposition)
        current = nodeList[(currentposition[0], currentposition[1])][currentposition[2]]
        for a in actions:
            # check collision
            newposition = (currentposition[0] + a[0], currentposition[1] + a[1])
            if collisionCheck(riskMap, newposition) == False:
                #print(currentposition)
                newsafety = currentposition[2] * (1 - riskMap[newposition[0]][newposition[1]])
                newsafety = round(newsafety, 3)
                if newsafety > safec:
                    newpositionwithsafety = (newposition[0], newposition[1], newsafety)
                    newg = current.g + getCostOfActionsEuclideanDistance(a)
                    if heuristic == 'manhattan':
                        newc = newg + manhattanHeuristic(newposition, xG)
                    if heuristic == 'euclidean':
                        newc = newg + euclideanHeuristic(newposition, xG)
                    if heuristic == 'learning':
                        newc = newg + learnedHeuristic(newposition, hmap, len(riskMap))
                    # check if new node found add to nodeList and pripority queue
                    if newposition not in nodeList:
                        newnode = node(newpositionwithsafety, newg, newc)
                        newnode.path = current.path.copy()
                        newnode.path.append(newpositionwithsafety)
                        nodeList[newposition] = {}
                        nodeList[newposition][newsafety] = newnode
                        visited.insert(newnode)
                    else:
                        approved = 0
                        for oldsafety, oldnode in nodeList[newposition].items():
                            # newnode is better in both, remove the old node
                            if newc < oldnode.gh and oldsafety < newsafety:
                                if oldnode.position in  visited.queue:
                                    visited.remove(oldnode.position)
                                approved = 1
                            # newnode is worse in distance but has better safety, keep both node
                            elif newc >= oldnode.gh and oldsafety < newsafety:
                                approved = 1 
                            # newnode is better in distance but has worse safety, keep both node
                            elif newc < oldnode.gh and oldsafety > newsafety:
                                approved = 1
                            #if there is an old node is better than the current node in both, abandon new node
                            elif newc >= oldnode.gh and oldsafety >= newsafety:
                                approved = 0
                                break
                        if approved:
                            newnode = node(newpositionwithsafety, newg, newc)
                            newnode.path = current.path.copy()
                            newnode.path.append(newpositionwithsafety)
                            nodeList[newposition][newsafety] = newnode
                            visited.insert(newnode)
                                
                        

    path = nodeList[(xG[0], xG[1])][xG[2]].path
    actionList = []
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        action = (node2[0] - node1[0], node2[1] - node1[1])
        if action == (-1, 0):
            action = 1
        elif action == (0, 1):
            action = 2
        elif action == (1, 0):
            action = 3
        elif action == (0, -1):
            action = 4
        actionList.append(action)
    return actionList, path, nodeList, count, explored