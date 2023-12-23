import networkx as nx
import time

from matrix import Matrix
from shortestPath import R, dijkstra, plt

def nearest_neighbor(graph, start_city):
    route = []
    route.append(start_city)
    visited = {}
    for city in graph.nodes:
        visited[city] = False

    visited[start_city] = True

    curr_city = start_city
    while not all(visited.values()):
        nearestc = None
        nearestdis = float('inf')

        for neighbor in graph.neighbors(curr_city): # Nearest Neighbor
            if not visited[neighbor]:
                distance = graph[curr_city][neighbor]["weight"]
                if distance < nearestdis:
                    nearestc = neighbor
                    nearestdis = distance

        if nearestc is not None:
            visited[nearestc] = True
            route.append(nearestc)
            curr_city = nearestc
        else:
            break

    route.append(start_city)

    totaldis = 0

    if not all(visited.values()):
        return route[:-1], "Not finished route"

    for i in range(len(route)-1):
        curr_city = route[i]
        next_city = route[i+1]
        distance_bet = graph[curr_city][next_city]["weight"]
        totaldis += distance_bet

    return route, totaldis


def BranchAndBound(graph, start_city):
    # Break up problem in sub problems
    lower_bound = float('-inf')
    upper_bound = float('inf')
    # create initial cost matrix
    initial_matrix = []
    neighbors = list(graph.neighbors(start_city))
    neighbors.insert(0, start_city)
    for x in range(0, len(neighbors)):
        row = []
        for y in range(0, len(neighbors)):
            if x == y:
                row.append(float('inf'))
            else:
                row.append(graph[neighbors[x]][neighbors[y]]['weight'])
        initial_matrix.append(row)
    start_matrix = Matrix(initial_matrix, None, 0)
    cost = ReduceMatrix(start_matrix)
    start_matrix.cost = cost
    lower_bound = cost
    fringe = [start_matrix]
    while len(fringe) != 0:
        currentMatrix = fringe.pop(0)
        for x in range(0, len(neighbors)):
            if x != currentMatrix.index and x not in currentMatrix.pastIndex:
                neighborMatrix = Matrix(currentMatrix.values, currentMatrix, x)
                neighborMatrix.setRowInf(neighborMatrix.parent.index)
                neighborMatrix.setColInf(neighborMatrix.index)
                cost = ReduceMatrix(neighborMatrix)
                if neighborMatrix.parent is not None:
                    neighborMatrix.cost = \
                    graph[neighbors[neighborMatrix.parent.index]][neighbors[neighborMatrix.index]][
                        'weight'] + neighborMatrix.parent.cost + cost
                currentMatrix.children.append(neighborMatrix)
                fringe.append(neighborMatrix)

        if len(currentMatrix.children) == 0:
            upper_bound = currentMatrix.cost
            for x in fringe:
                if x.cost >= upper_bound:
                    fringe.remove(x)
        fringe.sort(key=lambda x: x.cost)
        if len(fringe) == 0:
            return currentMatrix


def ReduceMatrix(matrix):
    cost = 0
    rowMin = matrix.rowMinimums()
    for x in range(0, len(matrix.values)):
        cost += rowMin[x]
        for y in range(0, len(matrix.values[x])):
            if matrix.values[x][y] != float('inf'):
                matrix.values[x][y] -= rowMin[x]

    colMin = matrix.colMinimums()
    for x in range(0, len(matrix.values[0])):
        cost += colMin[x]
        for y in range(0, len(matrix.values)):
            if matrix.values[y][x] != float('inf'):
                matrix.values[y][x] -= colMin[x]

    return cost


# BruteForce Method
# 1 city as starting point
# Generate all (n-1)! permutations of cities
# calculate cost of every permutation and keep track of min. cost permutation
# return permutation with minimal cost
# return shortest route and distance

import itertools


# matrix to hold all distances

def distancematrix(graph):
    nodes = list(graph.nodes)
    distancem = {}
    for src in nodes:
        dist, _ = dijkstra(graph, src)
        distancem[src] = dist
    return distancem


# total distance covered
def totdis(route, distancem):
    total = 0
    for i in range(len(route)):
        precity = route[i - 1]
        currcity = route[i]
        distance = distancem[precity][currcity]
        total += distance
    return total


def bruteforce(graph, distancem, start_city):
    locations = list(graph.nodes)
    locations.remove(start_city)  #removes start city from list
    shortest = None
    shortdist = float('inf')

    for route in itertools.permutations(locations):
        route = (start_city,) + route + (start_city,)  # start city first in route
        routedist = totdis(route + (start_city,),
                           distancem)  # start city end of route to finish loop.
        if routedist < shortdist:
            shortest = route
            shortdist = routedist

    return shortest, shortdist

def get_distance(path):
    current = 0
    distance = 0
    while current < len(path)-1:
        distance += R[path[current]][path[current+1]]["weight"]
        current += 1
    return distance
  
starting_city = "a"

start_time1 = time.time()

solutionMatrix = BranchAndBound(R, starting_city)
time1 = "--- %s seconds ---" % (time.time() - start_time1)


neighborsO = list(R.neighbors(starting_city))
neighborsO.insert(0, starting_city)
path = []
for x in solutionMatrix.pastIndex:
    path.insert(0, neighborsO[x])
path.append(neighborsO[solutionMatrix.index])
path.append(starting_city)
result = ""
BNB_distance = get_distance(path)
for x in range(0, len(path)):
    if x == len(path)-1:
        result += path[x]
    else:
        result += path[x] + " --- "
print("BAB Route = " + result)
print("Shortest Distance: ", BNB_distance)
print(time1)
print()

start_time2 = time.time()
distance_matrix = distancematrix(R)
shortest_route, shortest_distance = bruteforce(R, distance_matrix, starting_city)


print("Shortest Route:", " --- ".join(shortest_route))
print("Shortest Distance:", shortest_distance)
print("--- %s seconds ---" % (time.time() - start_time2))
print()

start_time3 = time.time()

start_city = "a"
route, distance = nearest_neighbor(R, start_city)
print("Route:", " --- ".join(route))
print("Total distance:", distance)
print("--- %s seconds ---" % (time.time() - start_time3))
plt.show()
