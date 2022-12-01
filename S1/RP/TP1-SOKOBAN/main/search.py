from sokoPuzzle import SokoPuzzle
from node import Node
from collections import deque
import itertools
import numpy as np
import sokoPuzzleGraphics


class Search:
    """ Uninformed/Blind Search """

    @staticmethod
    def breadthFirst(initial_node):

        # Check if the start element is the goal
        if initial_node.state.isGoal(Node.wall_space_obstacle):
            return initial_node, 0

        # Create the OPEN FIFO queue and the CLOSED list
        open = deque([initial_node])  # A FIFO queue
        closed = list()

        step = 0
        while True:

            # print (f'*** Step {step} ***')
            step += 1

            # Check if the OPEN queue is empty => goal not found 
            if len(open) == 0:
                return None, -1

            # Get the first element of the OPEN queue
            current = open.popleft()

            # Put the current node in the CLOSED list
            closed.append(current)

            # Generate the successors of the current node
            succ = current.succ()
            while len(succ) != 0:
                child = succ.popleft()

                # Check if the child is not in the OPEN queue and the CLOSED list
                if (child.state.robot_block not in [n.state.robot_block for n in
                                                    closed] and child.state.robot_block not in [n.state.robot_block for
                                                                                                n in open]):

                    # Put the child in the OPEN queue 
                    open.append(child)

                    # Check if the child is the goal
                    if child.state.isGoal(Node.wall_space_obstacle):
                        print(f'*** Step {step} ***')
                        print("Goal reached")
                        return child, step

    @staticmethod
    def A(initial_node, heuristic=1):

        # Check if the initial node is the goal
        if initial_node.state.isGoal(Node.wall_space_obstacle):
            return initial_node, 0

        # Evaluate the cost f for the initial node
        initial_node.F_Evaluation(heuristic)

        # Create the OPEN list with the initial node as the first element
        open = [initial_node]

        # Create the CLOSED list
        closed = list()
        deadlock = list()

        step = 0
        while True:

            step += 1
            # print (f'*** Step {step} ***')

            # Check if the OPEN list is empty => goal not found 
            if len(open) == 0:
                return None, -1

            # Get the index of the node with least f in the OPEN list 
            min_index, _ = min(enumerate(open), key=lambda element: element[1].f)
            current = open[min_index]

            # Remove the current node (i.e. the node with least f) from the OPEN list
            open.remove(current)

            # Put the current node in the CLOSED list
            closed.append(current)

            # Check if the current state is a goal
            if current.state.isGoal(Node.wall_space_obstacle):
                print("Goal reached")
                return current, step

                # Generate the successors of the current node
            succ = current.succd()
            while len(succ) != 0:
                # Pop a child node from the list of successors
                child = succ.popleft()
                # Evaluate the cost f for this child node
                child.F_Evaluation(heuristic)

                # If the child is in the OPEN list
                if child.state.robot_block in [n.state.robot_block for n in open]:
                    # Get the index of the child in the OPEN list
                    index = [n.state.robot_block for n in open].index(child.state.robot_block)
                    # Replace the node in the OPEN list by the new one if its cost f is less than the old one
                    if open[index].f > child.f:
                        # Remove the old node from the OPEN list
                        open.remove(open[index])
                        # Put the new node with the minimal cost f in the OPEN list 
                        open.append(child)

                        # If the child is not in the OPEN list
                else:
                    # If the child is not in the CLOSED list
                    if child.state.robot_block not in [n.state.robot_block for n in closed]:
                        # Put the child in the OPEN list 
                        open.append(child)

                        # If the child is in the CLOSED list
                    else:
                        # Get the index of the child in the CLOSED list
                        index = [n.state.robot_block for n in closed].index(child.state.robot_block)
                        # Remove the node from the CLOSED list and add the new one in the OPEN list if its cost f is less than the old one
                        if closed[index].f > child.f:
                            # Remove the child from the CLOSED list
                            closed.remove(closed[index])
                            # Put the child in the OPEN list 
                            open.append(child)

    @staticmethod
    def AA(initial_node, heuristic=1):

        # Check if the initial node is the goal
        if initial_node.state.isGoal(Node.wall_space_obstacle):
            return initial_node, 0

        # Evaluate the cost f for the initial node
        initial_node.F_Evaluation(heuristic)

        # Create the OPEN list with the initial node as the first element
        open = [initial_node]

        # Create the CLOSED list
        closed = list()

        step = 0
        while True:

            step += 1
            # print (f'*** Step {step} ***')

            # Check if the OPEN list is empty => goal not found 
            if len(open) == 0:
                return None, -1

            # Get the index of the node with least f in the OPEN list 
            min_index, _ = min(enumerate(open), key=lambda element: element[1].f)
            current = open[min_index]

            # Remove the current node (i.e. the node with least f) from the OPEN list
            open.remove(current)

            # Put the current node in the CLOSED list
            closed.append(current)

            # Check if the current state is a goal
            if current.state.isGoal(Node.wall_space_obstacle):
                print("Goal reached")
                return current, step

                # Generate the successors of the current node
            succ = current.succd()
            while len(succ) != 0:
                # Pop a child node from the list of successors 
                child = succ.popleft()
                # Evaluate the cost f for this child node
                child.F_Evaluation(heuristic)

                # If the child is in the OPEN list
                if child.state.robot_block in [n.state.robot_block for n in open]:
                    # Get the index of the child in the OPEN list
                    index = [n.state.robot_block for n in open].index(child.state.robot_block)
                    # Replace the node in the OPEN list by the new one if its cost f is less than the old one
                    if open[index].f > child.f:
                        # Remove the old node from the OPEN list
                        open.remove(open[index])
                        # Put the new node with the minimal cost f in the OPEN list 
                        open.append(child)

                        # If the child is not in the OPEN list
                else:
                    # If the child is not in the CLOSED list
                    if child.state.robot_block not in [n.state.robot_block for n in closed]:
                        # Put the child in the OPEN list 
                        open.append(child)

                        # If the child is in the CLOSED list
                    else:
                        # Get the index of the child in the CLOSED list
                        index = [n.state.robot_block for n in closed].index(child.state.robot_block)
                        # Remove the node from the CLOSED list and add the new one in the OPEN list if its cost f is less than the old one
                        if closed[index].f > child.f:
                            # Remove the child from the CLOSED list
                            closed.remove(closed[index])
                            # Put the child in the OPEN list 
                            open.append(child)


""" ***************************************************** Main function **************************************************** """

board1 = [['O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'S', ' ', 'B', ' ', 'O'],
          ['O', ' ', 'O', 'R', ' ', 'O'],
          ['O', ' ', ' ', ' ', ' ', 'O'],
          ['O', ' ', ' ', ' ', ' ', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O']]

board2 = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
          ['O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
          ['O', ' ', ' ', 'O', 'O', 'O', ' ', ' ', 'O'],
          ['O', ' ', ' ', ' ', ' ', 'O', '.', ' ', 'O'],
          ['O', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'O'],
          ['O', ' ', ' ', 'B', ' ', ' ', 'O', ' ', 'O'],
          ['O', ' ', ' ', ' ', ' ', ' ', 'O', ' ', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

board3 = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', ' ', ' ', ' ', 'O', ' ', ' ', 'O'],
          ['O', ' ', ' ', 'B', 'R', ' ', ' ', 'O'],
          ['O', ' ', ' ', ' ', 'O', 'B', ' ', 'O'],
          ['O', 'O', 'O', 'O', 'O', ' ', 'S', 'O'],
          ['O', 'O', 'O', 'O', 'O', ' ', 'S', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

board4 = [['O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'O', ' ', ' ', 'O', 'O', 'O'],
          ['O', 'O', ' ', ' ', 'O', 'O', 'O'],
          ['O', 'O', ' ', '*', ' ', ' ', 'O'],
          ['O', 'O', 'B', 'O', 'B', ' ', 'O'],
          ['O', ' ', 'S', 'R', 'S', ' ', 'O'],
          ['O', ' ', ' ', ' ', ' ', 'O', 'O'],
          ['O', 'O', 'O', ' ', ' ', 'O', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O']]

board5 = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'S', 'O', ' ', ' ', 'O', 'O'],
          ['O', ' ', ' ', ' ', ' ', 'B', ' ', 'O', 'O'],
          ['O', ' ', 'B', ' ', 'R', ' ', ' ', 'S', 'O'],
          ['O', 'O', 'O', ' ', 'O', ' ', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'B', 'O', ' ', 'O', 'O', 'O'],
          ['O', 'O', 'O', ' ', ' ', 'S', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

boardTEST = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', 'O', 'O', 'O', 'O', ' ', ' ', ' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', 'O', 'O', 'O', 'O', 'B', ' ', ' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', 'O', 'O', 'O', 'O', ' ', ' ', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', 'O', 'O', ' ', ' ', 'B', ' ', 'B', ' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', 'O', 'O', ' ', 'O', ' ', 'O', 'O', ' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', ' ', ' ', ' ', 'O', ' ', 'O', 'O', ' ', 'O', 'O', 'O', 'O', 'O', ' ', ' ', 'S', 'S', 'O'],
             ['O', ' ', 'B', ' ', ' ', 'B', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'S', 'S', 'O'],
             ['O', 'O', 'O', 'O', 'O', ' ', 'O', 'O', 'O', ' ', 'O', 'R', 'O', 'O', ' ', ' ', 'S', 'S', 'O'],
             ['O', 'O', 'O', 'O', 'O', ' ', ' ', ' ', ' ', ' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
             ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

board6 = [['O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'S', ' ', 'O', ' ', 'R', 'O'],
          ['O', ' ', ' ', 'O', 'B', ' ', 'O'],
          ['O', 'S', ' ', ' ', 'B', ' ', 'O'],
          ['O', ' ', ' ', 'O', 'B', ' ', 'O'],
          ['O', 'S', ' ', 'O', ' ', ' ', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O']]

board7 = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'S', 'S', 'S', ' ', 'O', 'O', 'O'],
          ['O', ' ', 's', ' ', 'B', ' ', ' ', 'O'],
          ['O', ' ', ' ', 'B', 'B', 'B', ' ', 'O'],
          ['O', 'O', 'O', 'O', ' ', ' ', 'R', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

board8 = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', ' ', ' ', ' ', ' ', 'O', 'O', 'O'],
          ['O', ' ', ' ', ' ', 'B', ' ', ' ', 'O'],
          ['O', 'S', 'S', 'S', '*', 'B', 'R', 'O'],
          ['O', ' ', ' ', ' ', 'B', ' ', ' ', 'O'],
          ['O', ' ', ' ', ' ', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

board9 = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'S', ' ', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', ' ', ' ', 'O', 'O', 'O', 'O'],
          ['O', 'S', ' ', 'S', ' ', 'O', 'O', 'O', 'O'],
          ['O', ' ', 'B', ' ', 'B', 'B', ' ', ' ', 'O'],
          ['O', 'O', 'O', 'S', ' ', ' ', 'B', 'R', 'O'],
          ['O', 'O', 'O', ' ', ' ', 'O', 'O', 'O', 'O'],
          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]


def create_initial_node(board=None):
    global indexV1, indexV2, nbH, indexH1, indexH2
    height = len(board)
    width = len(board[0])
    # Separate walls, spaces and obstacles board from robot and boxes board
    robot_block = [[''] * width for _ in range(height)]
    wall_space_obstacle = [[''] * width for _ in range(height)]
    deadlock_matrice = np.array([[' '] * width for _ in range(height)])  # creer matrice deadlock initialemnt vide

    for i, j in itertools.product(range(height), range(width)):
        if board[i][j] == 'R':
            robot_position = (i, j)
            robot_block[i][j] = 'R'
            wall_space_obstacle[i][j] = ' '
        elif board[i][j] == 'B':
            robot_block[i][j] = 'B'
            wall_space_obstacle[i][j] = ' '
        elif board[i][j] == 'S' or board[i][j] == 'O' or board[i][j] == ' ':
            robot_block[i][j] = ' '
            wall_space_obstacle[i][j] = board[i][j]
        elif board[i][j] == '*':
            robot_block[i][j] = 'B'
            wall_space_obstacle[i][j] = 'S'
        else:  # self.board[i][j] == '.'
            robot_position = (i, j)
            robot_block[i][j] = 'R'
            wall_space_obstacle[i][j] = 'S'

        # deadlock coin
        if board[i][j] == ' ' or board[i][j] == 'R':
            if ((board[i - 1][j] == 'O' and board[i][j - 1] == 'O') or (
                    board[i - 1][j] == 'O' and board[i][j + 1] == 'O') or (
                    board[i][j - 1] == 'O' and board[i + 1][j] == 'O')
                    or (board[i][j + 1] == 'O' and board[i + 1][j]) == 'O'):
                deadlock_matrice[i][j] = 'D'
                board[i][j] = 'D'

    for i1 in range(height):
        list = deque()
        ligne_mur_haut = True
        ligne_mur_bas = True
        storage = 0
        for j1 in range(width):
            if (deadlock_matrice[i1][j1] == 'D'):
                list.append((i1, j1))
            # if (wall_space_obstacle[i1][j1]=='S'):
            # storage=storage+1

        # if storage==0:
        len_listt = len(list)
        if len_listt >= 2:

            if (len(list) % 2) == 0:
                len_list = len(list)
            else:
                len_list = len(list) - 1

            for s in range(len_list - 1):
                if len(list) >= 2:
                    x0, y0 = list.popleft()
                    x1, y1 = list.popleft()

                    for x in range(y0, y1):

                        if wall_space_obstacle[i1 - 1][x] != 'O' or wall_space_obstacle[i1][x] == 'S':
                            ligne_mur_haut = False

                        if wall_space_obstacle[i1 + 1][x] != 'O' or wall_space_obstacle[i1][x] == 'S':
                            ligne_mur_bas = False

                    if (ligne_mur_bas != False):
                        for x1 in range(y0, y1):
                            if (wall_space_obstacle[i1][x1] == ' ' or robot_block[i1][x1] == 'R'):
                                deadlock_matrice[i1][x1] = 'D'
                                board[i1][x1] = 'D'

                    if (ligne_mur_haut != False):
                        for x2 in range(y0, y1):
                            if (wall_space_obstacle[i1][x2] == ' ' or robot_block[i1][x2] == 'R'):
                                deadlock_matrice[i1][x2] = 'D'
                                board[i1][x2] = 'D'

    # deadlock matrice de coté des column
    for j in range(width):
        listC = deque()
        ligne_mur_right = True
        ligne_mur_left = True
        for i in range(height):
            if (deadlock_matrice[i][j] == 'D'):
                listC.append((i, j))

        len_list1 = len(listC)
        if len_list1 >= 2:

            if (len(listC) % 2) == 0:
                len_list = len(listC)
            else:
                len_list = len(listC) - 1

            for s in range(len_list - 1):
                if len(listC) >= 2:
                    x0, y0 = listC.popleft()
                    x1, y1 = listC.popleft()

                    for x in range(x0, x1):

                        if wall_space_obstacle[x][j - 1] != 'O' or wall_space_obstacle[x][j] == 'S':
                            ligne_mur_right = False

                        if wall_space_obstacle[x][j + 1] != 'O' or wall_space_obstacle[x][j] == 'S':
                            ligne_mur_left = False

                    if (ligne_mur_left != False):
                        for yl in range(x0, x1):
                            if (wall_space_obstacle[yl][j] == ' ' or robot_block[yl][j] == 'R'):
                                deadlock_matrice[yl][j] = 'D'
                                board[yl][j] = 'D'

                    if (ligne_mur_right != False):
                        for yr in range(x0, x1):
                            if (wall_space_obstacle[yr][j] == ' ' or robot_block[yr][j] == 'R'):
                                deadlock_matrice[yr][j] = 'D'
                                board[yr][j] = 'D'

    # matrice deadlock only
    with open('deadlockmatrice.txt', 'w') as d:
        d.write(str(deadlock_matrice))
    # new map with deadlock
    for row, col in itertools.product(range(height), range(width)):
        if deadlock_matrice[row][col] == 'D':
            board[row][col] = 'D'
    with open('newMap.txt', 'w') as d:
        d.write(str(np.array(board)))
    Node.wall_space_obstacle = wall_space_obstacle
    Node.deadlock_matrice = deadlock_matrice
    initial_node = Node(SokoPuzzle(robot_block, robot_position))
    return initial_node


levels = [board1, board2, board3, board4, board5, board6]
solutions = []

# for game in levels:
# level = game
level = board3
initial_node = create_initial_node(board=level)

# utilisation de A
goalNode, num_steps = Search.A(initial_node, heuristic=3)
if goalNode:
    print(f"Optimal Solution found after {num_steps} steps")
    solution = goalNode.getSolution()
    solutions.append(solution)
    print(f"nombre de mouvements:{len(solution) - 1}")

    # ecrire le résultat obtenu dans un fichier text
    """with open('README.txt', 'a') as d:
            d.write(f"Pour le board il a fallu {num_steps} itérations\n")"""
else:
    print("Optimal solution not found")


# utilisation de BFS
"""goalNode, num_steps = Search.breadthFirst(initial_node)
if goalNode:
    print(f"Optimal Solution found after {num_steps} steps")
    solution = goalNode.getSolution()
    with open('README.txt', 'a') as d:
        d.write(f"Pour le board il a fallu {num_steps} itérations\n")

else:
    print("Optimal solution not found")"""
