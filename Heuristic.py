import bfsdfs

straightline_Bucharest= {'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Dobreta': 242, 'Eforie': 161,
                         'Yagaras': 178, 'Giurgiu':	77, 'Hirsova':	151, 'Iasi': 226, 'Lugoj': 244,
                         'Mehadia': 241, 'Neamt': 234, 'Oradea': 380, 'Pitesti': 98,'Rimnicu_Vilcea': 193,
                         'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind':	374}

def checknode(node, nodes =[]):
    global totalcost
    for i in range(len(nodes)):
        if node == nodes[i][0]:
            bfsdfs.totalcost += nodes[i][1] #totalcost 계산
            return True
    return False

def trace(start, closed= [], path =[]):
    global totalcost
    global open
    node = start
    for i in range(len(closed)):
        check = checknode(node, bfsdfs.mapping[closed[i]])  #closed에 있는 도시가 연결이 되어있는 도시를 검색, 해당 노드의 도시가 있다면 True 없다면 False를 리턴
        if check is True:
            path.insert(0, node)
            node = closed[i]    #노드에 검색했던 도시를 넣고 closed의 그 다음 노드부터 계속 탐색해나간다
    path.insert(0, node)
    print("path :", path)
    print("totalcost : ", bfsdfs.totalcost)
    print("open : ", bfsdfs.open)
    print("closed : ", closed)  #탐색이 종료되고 난 후 결과를 프린트한다


def sortbytuple(nodes =[], currentcost =0, open =[], closed =[]):
    temp =[]
    for i in range(len(nodes)):
        duplication = False     #indicator를 초기화해주고
        if nodes[i][0] not in closed:
            if len(open) != 0:
                for n in range(len(open)):
                    if nodes[i][0] in open[n]:
                        duplication = True  #겹치는게 있을 시 indicator에 표시
                        break
                    else:
                        duplication = False
        else:
            duplication = True
        if duplication is False:
            # 겹치는 노드가 없다면 리턴할 List에 경로 추가 이때, tuple의 형식에 맞게 현재까지 온 거리와, 평가값 계산
            temp.append((nodes[i][0], currentcost + nodes[i][1], currentcost + nodes[i][1] + straightline_Bucharest[nodes[i][0]]))
    return temp

def best_first(start, goal, tupleopen =[], closed =[]):
    if len(closed) == 0:
        tupleopen.append((start, 0, straightline_Bucharest[start])) #처음에 시작 노드를 추가해준다 각각 노드이름, 현재까지 온 총 거리 g*(n), 현재까지 온 총 거리에 더하는 직선거리 f(n)
    bfsdfs.printprocess(tupleopen, closed)  #진척사항을 프린트해준다
    element = tupleopen.pop(0)  #open에서 첫번째꺼 하나뽑아
    closed.insert(0, element[0]) #closed에 넣어준다음
    sortedArr = sortbytuple(bfsdfs.mapping[element[0]], element[1], tupleopen, closed)  #다음으로 갈 노드들 중 중복되는 장소를 거른다
    for i in range(len(sortedArr)):
        bfsdfs.nodecount += 1
        if sortedArr[i][0] == goal: #그 장소들을 들여다 봤을 때 목적지가 있으면
            global totalcost
            path =[]
            path.insert(0, goal)
            bfsdfs.totalcost = 0
            for n in range(len(bfsdfs.mapping[start])):
                if goal in bfsdfs.mapping[start][n]:
                    bfsdfs.totalcost += bfsdfs.mapping[start][n][1] #필요한 자료들을 초기화시켜주고
            print("best_first")
            trace(start, closed, path)  #추적한다
            print("nodecount ", bfsdfs.nodecount)   #nodecount 결과를 프린트해준다
            bfsdfs.nodecount = 0
            tupleopen.clear()
            closed.clear()  #리턴하기전 값들을 정리해준다
            return
        else:
            tupleopen.insert(0, sortedArr[i])
    temp2 = sorted(tupleopen, key=lambda tupleopen: tupleopen[2])   #f(n)을 기준으로 평가해서 sort하기 (evaluation)
    tupleopen = temp2
    element = tupleopen[0][0]   #다음에 탐색할 가장 첫번째 노드를 지정해주고
    best_first(element, goal, tupleopen, closed)    #탐색 과정을 반복한다
    return


def hill_climbing(start, goal, tupleopen =[], closed =[]):
    if len(closed) == 0:
        tupleopen.append((start, 0, straightline_Bucharest[start])) #처음에 시작 노드를 추가해준다 각각 노드이름, 현재까지 온 총 거리 g(n), 현재까지 온 총 거리에 더하는 직선거리 f(n)
    bfsdfs.printprocess(tupleopen, closed)  #진척사항을 프린트해준다
    element = tupleopen.pop(0)  #open에서 첫번째꺼 하나뽑아
    closed.insert(0, element[0])    #closed에 넣어준다음
    sortedArr = sortbytuple(bfsdfs.mapping[element[0]], element[1], tupleopen, closed)  #다음으로 갈 노드들 중 중복되는 장소를 거른다
    for i in range(len(sortedArr)):
        bfsdfs.nodecount += 1
        if sortedArr[i][0] == goal: #그 장소들을 들여다 봤을 때 목적지가 있으면
            global totalcost
            path =[]
            path.insert(0, goal)
            bfsdfs.totalcost = 0
            for n in range(len(bfsdfs.mapping[start])):
                if goal in bfsdfs.mapping[start][n]:
                    bfsdfs.totalcost += bfsdfs.mapping[start][n][1] #필요한 자료들을 초기화시켜주고
            print("hill_climbing")
            trace(start, closed, path)  #추적한다
            print("nodecount ", bfsdfs.nodecount)   #nodecount 결과를 프린트해준다
            bfsdfs.nodecount = 0
            tupleopen.clear()
            closed.clear()  #리턴하기전 값들을 정리해준다
            return
    refinedArr = sorted(sortedArr, key=lambda sortedArr: sortedArr[2])  #f(n)을 기준으로 평가해서 sort하기 (evaluation)
    refinedArr.reverse()    #역순을 취해서 가장 낮은 평가값이 맨 앞에 오도록 삽입해준다
    for i in range(len(sortedArr)):
        tupleopen.insert(0, refinedArr[i])
    element = tupleopen[0][0]   #다음에 탐색할 가장 첫번째 노드를 지정해주고
    hill_climbing(element, goal, tupleopen, closed) #탐색 과정을 반복한다
    return


hill_climbing('Timisoara', 'Bucharest', bfsdfs.open, bfsdfs.closed)
print("\n")
best_first('Timisoara', 'Bucharest', bfsdfs.open, bfsdfs.closed)
