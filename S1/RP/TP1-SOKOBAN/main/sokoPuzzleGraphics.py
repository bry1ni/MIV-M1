import search
import pygame
import time

import images


wall = pygame.image.load('images/block_05.png')
BOX_SIZE = wall.get_width()

clock = pygame.time.Clock()

pygame.font.init()

board = search.level


# Dessiner l'interface par rapport au board
def drawBoard(screen, board):
    storagePoint = pygame.image.load('images/environment_06.png')
    box = pygame.image.load('images/crate_07.png')
    robot = pygame.image.load('images/player_22.png')
    space = pygame.image.load('images/square.png')
    boxIN = pygame.image.load('images/crate_27.png')
    robIN = pygame.image.load('images/player_03.png')
    deadlock = pygame.image.load('images/environment_05.png')

    images = {
        'O': wall,
        'S': storagePoint,
        'B': box,
        'R': robot,
        ' ': space,
        '*': boxIN,
        '.': robIN,
        'D': deadlock
    }
    screen.fill("white")
    # pygame.display.flip()
    for i in range(len(board)):
        for j in range(len(board[i])):
            screen.blit(images[board[i][j]], (j * BOX_SIZE, i * BOX_SIZE))
    pygame.display.update()


# Affichage d'un texte sur l'interfac
def text_objects(text, font):
    textSurface = font.render(text, True, (0, 0, 0))
    return textSurface, textSurface.get_rect()


def message_display(text, size, height, width):
    h, w = screenSize()
    largeText = pygame.font.Font('Blackout.otf', size)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = (int(height), int(width))
    screen.blit(TextSurf, TextRect)
    pygame.display.update()
    time.sleep(1)


solution = search.solution
solutions = search.solutions


# calculer la taille de l'interface par rapport au board
def screenSize():
    j = 0
    for i in range(len(board)):
        j = len(board[i]) if len(board[i]) > j else j

    return j * BOX_SIZE, len(board) * BOX_SIZE


# -------------------- intialisation de pygame ------------------------
screen = pygame.display.set_mode(screenSize())
pygame.display.set_caption("Sokoban MIV - M1")
screen.fill((0, 0, 0))
done = False

h, w = screenSize()

while not done:

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
            done = True

    # mettre a jour l'interface selon la matrice solution
    for i in range(len(solution)):
        drawBoard(screen, solution[i])
        time.sleep(0.5)
        pygame.display.flip()

    # affichage de nombre ditération sur l'interface
    message_display(f"{len(solution) - 1} moves", 65, h / 2, w / 2)
    message_display(f"{search.num_steps} iterations", 25, h / 2, w / 1.7)
    pygame.display.flip()
    clock.tick(120)
    done = True
