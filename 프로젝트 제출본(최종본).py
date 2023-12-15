from collections import deque

class MazeGraph:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.vertices = []
        self.adjList = {}

    def insertVertex(self, v):
        self.vertices.append(v)
        self.adjList[v] = []

    def insertEdge(self, u, v):
        if v not in self.adjList[u]:
            self.adjList[u].append(v)
        if u not in self.adjList[v]:
            self.adjList[v].append(u)

    def buildGraph(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == '0':
                    vertex = (i, j)
                    self.insertVertex(vertex)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == '0':
                    vertex = (i, j)
                    neighbors = self.getNeighbors(i, j)
                    for neighbor in neighbors:
                        self.insertEdge(vertex, neighbor)

    def getNeighbors(self, i, j):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dir in directions:
            ni, nj = i + dir[0], j + dir[1]
            if 0 <= ni < self.rows and 0 <= nj < self.cols and self.maze[ni][nj] == '0':
                neighbors.append((ni, nj))
        return neighbors
    def shortestPath(self, start, end):
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            visited.add(current)
            for neighbor in self.adjList[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None

    def printPath(self, path):
        print("경로:", end=" ")
        for i, vertex in enumerate(path):
            if i == 0:
                print("e", end=" – ")
            elif i == len(path) - 1:
                print("x", end="")
            else:
                vertex_num = self.vertices.index(vertex)
                print(f"{vertex_num} –", end=" ")

maze = [
    ['1', '1', '1', '1', '1', '1'],
    ['0', '0', '0', '0', '0', '1'],
    ['1', '0', '1', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '1'],
    ['1', '0', '1', '0', '0', '0'],
    ['1', '0', '0', '0', '1', '1'],
    ['1', '1', '1', '1', '1', '1']
]

graph = MazeGraph(maze)

graph.buildGraph()

start_vertex = (1, 0)  # 정점e
end_vertex = (4, 5)    # 정점x
shortest_path = graph.shortestPath(start_vertex, end_vertex)

if shortest_path:
    print("정점:", graph.vertices)
    print("인접리스트 :")
    i=1
    for vertex in graph.vertices:
        if (vertex == start_vertex):
            print(f"e: {graph.adjList[vertex]}")
            continue;
        if (vertex == end_vertex):
            print(f"x: {graph.adjList[vertex]}")
            continue;
        print(f"{i}: {graph.adjList[vertex]}")
        i+=1;
    graph.printPath(shortest_path)
    print("\n길이:", len(shortest_path) - 1)