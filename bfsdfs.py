open =[]
closed =[]
totalcost = 0
nodecount = 0

Arad = [('Timisoara', 118), ('Zerind', 75), ('Sibiu', 140)]
Bucharest = [('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85), ('Yagaras', 211)]
Craiova = [('Dobreta', 120), ('Rimnicu_Vilcea', 146), ('Pitesti', 138)]
Dobreta = [('Mehadia', 75), ('Craiova', 120)]
Eforie = [('Hirsova', 86)]
Yagaras = [('Sibiu', 99), ('Bucharest', 211)]
Giurgiu = [('Bucharest', 90)]
Hirsova = [('Eforie', 86), ('Urziceni', 98)]
Iasi = [('Neamt', 87), ('Vaslui', 92)]
Lugoj = [('Timisoara', 111), ('Mehadia', 70)]
Mehadia = [('Lugoj', 70), ('Dobreta', 75)]
Neamt = [('Iasi', 87)]
Oradea = [('Zerind', 71), ('Sibiu', 151)]
Pitesti = [('Rimnicu_Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)]
Rimnicu_Vilcea = [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)]
Sibiu = [('Arad', 140), ('Oradea', 151), ('Yagaras', 99), ('Rimnicu_Vilcea', 80)]
Timisoara = [('Arad', 118), ('Lugoj', 111)]
Urziceni = [('Bucharest', 85), ('Hirsova', 98), ('Vaslui', 142)]
Vaslui = [('Iasi', 92), ('Urziceni', 142)]
Zerind = [('Arad', 75), ('Oradea', 71)]

mapping = {'Arad': Arad, 'Bucharest': Bucharest, 'Craiova': Craiova, 'Dobreta': Dobreta, 'Eforie': Eforie, 'Yagaras': Yagaras, 'Giurgiu': Giurgiu, 'Hirsova': Hirsova, 'Iasi': Iasi, 'Lugoj': Lugoj, 'Mehadia': Mehadia, 'Neamt': Neamt, 'Oradea': Oradea, 'Pitesti': Pitesti, 'Rimnicu_Vilcea': Rimnicu_Vilcea, 'Sibiu': Sibiu, 'Timisoara': Timisoara, 'Urziceni': Urziceni, 'Vaslui': Vaslui, 'Zerind': Zerind}



class paths:
    pathbag = [[]]
    pathstotalcost = []
    nodecount = []
    def insertpath(self, path=[], totalcost = 0, nodecount = 0):
        #pathbag에 경로를 저장, totalcost와 nodecount도 순서에 맞게 저장
        self.pathbag.insert(0,path)
        self.pathstotalcost.insert(0, totalcost)
        self.nodecount.insert(0, nodecount)
    def printshortestpath(self):
        #거리를 비교하여 가장 짧은 경로를 찾아 프린트
        tmpnum = 0
        for i in range(len(self.pathbag)-1):
            if self.pathstotalcost[tmpnum] >= self.pathstotalcost[i]:
               tmpnum = i
        print("shortest path : ", self.pathbag[tmpnum])
        print("totalcost : ", self.pathstotalcost[tmpnum])
        print("nodecount : ", self.nodecount[tmpnum])

def printprocess(open =[], closed =[]):
    print("open : ")
    print(open)
    print("closed : ")
    print(closed)
    print("\n")

def checkduplication(start, nodes=[], open= [], closed=[]):
    temp = []
    forreturn = []
    for i in range(len(closed)):
        temp.append(closed[i])
    for i in range(len(open)):  #closed와 open에 있는 모든 tuple들을 가져온다
        temp.append(open[i])
    for i in range(len(nodes)):
        duplication = False     #indicator를 초기화해주고
        for n in range(len(temp)):
            if nodes[i][0] in temp[n] and start in temp[n]:
                duplication = True  #겹치는게 있다면 indicator에 표시
        if duplication == False:
            forreturn.append(nodes[i][0]) #겹치는 노드가 없다면 리턴할 List에 경로 추가
    return forreturn

def checknode(node, nodes =[]):
    global totalcost
    for i in range(len(nodes)):
        if node == nodes[i][0]:
            totalcost += nodes[i][1]
            return True
    return False

def trace(start, goal, closed= [], path =[]):
    global totalcost
    global nodecount
    node = start
    for i in range(len(closed)):
        if closed[i][0] == node and node != goal:
            path.insert(0, node)
            for n in range(len(mapping[node])):
                if mapping[node][n][0] == closed[i][1]:
                    totalcost += mapping[node][n][1]
            node = closed[i][1]
    if node is "start":
        print("path :", path)
        print("totalcost : ", totalcost)
        print("open : ", open)
        print("closed : ", closed)
        print("nodecount ", nodecount)
        print("\n")
        tmp = paths()
        tmp.insertpath(path, totalcost, nodecount)

def bfs(start, goal, open=[], closed=[]):
    global nodecount
    global totalcost
    #처음 시작하는 도시를 open에 넣어준다
    if len(closed) == 0:
        open.append((start, "start"))
    printprocess(open, closed)  #진척된 사항을 표시한다
    if len(open) != 0:
        element = open.pop(0)   #open에서 첫 요소를 뺀 후에
        closed.insert(0, element)  #close에 그 요소를 추가한 다음
    # 도시와 연결된 곳 중에서 너비우선 법칙에 의해 정렬하여 이전에 가지 않았던 곳을 간다
    sortedArr = checkduplication(start, mapping[start], open, closed)  # 중복되는 도시 거르기
    refinedArr = sorted(sortedArr)  # 중복된 도시 거르고 난 후 알파벳 순으로 정렬
    for i in range(len(mapping[start])):    #주변에 목적지가 있는지 확인
        if mapping[start][i][0] == goal:    #주변에 목적지를 발견했다면
            path = []
            path.insert(0, goal)
            totalcost = 0
            for n in range(len(mapping[start])):
                if goal in mapping[start][n]:
                    totalcost += mapping[start][n][1]   #path, totalcost를 초기화시켜주고
            trace(start, goal, closed, path)    #추적을 시작한다
    if len(refinedArr) != 0:
        for i in range(len(refinedArr)):    #추적이 끝나거나 주변에 목적지가 없다면 가던 길을 계속 간다
            nodecount += 1
            open.append((refinedArr[i], start)) #정제된 도시 리스트를 open에 추가한다
    if len(open) == 0:  #모든 곳을 다 돌아봤다면
        result = paths()
        result.printshortestpath()
        result.pathbag.clear()
        open.clear()
        closed.clear()
        nodecount = 0   #모두 정리해주고 함수를 빠져나온다
        return
    element = open[0][0]    #모든 곳을 다 돌아본 것이 아니라면 open에서 뽑을 노드를 지정해주고
    bfs(element, goal, open, closed)    #같은 과정을 반복한다
    return


def dfs(start, goal, open=[], closed=[]):
    global nodecount
    global totalcost
    # 처음 시작하는 도시를 open에 넣어준다
    if len(closed) == 0:
        open.append((start, "start"))
    printprocess(open, closed)  # 진척된 사항을 표시한다
    # 도시와 연결된 곳 중에서 목적지가 있는지 확인한다
    if len(open) != 0:
        element = open.pop(0)  # open에서 첫 요소를 뺀 후에
        closed.insert(0, element)  # close에 그 요소를 추가한 다음
    # 도시와 연결된 곳 중에서 깊이우선 법칙에 의해 정렬하여 이전에 가지 않았던 곳을 간다
    sortedArr = checkduplication(start, mapping[start], open, closed)  # 중복되는 도시 거르기
    refinedArr = sorted(sortedArr)
    refinedArr.reverse()    #알파벳 역순으로 정렬해서 차례대로 넣어줘야 알파벳 순을 만족시킨다
    for i in range(len(mapping[start])):  # 주변에 목적지가 있는지 확인
        if mapping[start][i][0] == goal:
            path = []
            path.insert(0, goal)
            totalcost = 0
            for n in range(len(mapping[start])):
                if goal in mapping[start][n]:
                    totalcost += mapping[start][n][1]   #path, totalcost를 초기화시켜주고
            trace(start, goal, closed, path)    #추적을 시작한다
    if len(refinedArr) != 0:
        for i in range(0, len(refinedArr), 1):  #추적이 끝나거나 주변에 목적지가 없다면 가던 길을 계속 간다
            nodecount += 1
            open.insert(0, (refinedArr[i], start))  # 정제된 도시 리스트를 open에 추가한다
    if len(open) == 0:  #모든 곳을 다 돌아봤다면
        result = paths()
        result.printshortestpath()
        result.pathbag.clear()
        open.clear()
        closed.clear()
        nodecount = 0   #모두 정리해주고 함수를 빠져나온다
        return
    element = open[0][0]    #모든 곳을 다 돌아본 것이 아니라면 open에서 뽑을 노드를 지정해주고
    dfs(element, goal, open, closed)  # 같은 과정을 반복한다
    return

print("DFS")
dfs('Timisoara', 'Bucharest', open, closed)
print("\n")
print("BFS")
bfs('Timisoara', 'Bucharest', open, closed)
print("\n")
