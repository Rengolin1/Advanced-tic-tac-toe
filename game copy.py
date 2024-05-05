import pygame as pg
import sys
import numpy as np
import tensorflow as tf
import time
from typing import Any

# Инициализация Pygame
pg.init()
pg.font.init()
pg.mixer.init()

pg.mixer.music.load("BVG, møndberg - After Rain.mp3")
pg.mixer_music.set_volume(0.025)
pg.mixer.music.play()

Toe_sound = pg.mixer.Sound('cherchenie-ruchkoy-podpis.wav')
Tic_sound = pg.mixer.Sound('zavershayuschaya-cherta-pod-osnovnyim-tekstom.wav')

# Размеры окна
WIDTH, HEIGHT = 300, 300
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption('Крестики-нолики')

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
Q = False
# Игровое поле
board = [[None, None, None],
         [None, None, None],
         [None, None, None]]

# Размеры клеток
cell_size = 100
line_width = 5

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(3, 3)),
  tf.keras.layers.Dense(9, activation='relu'),
  tf.keras.layers.Dense(9)
])
def draw_board():
    screen.fill(BLACK)
    for x in range(1, 3):
        pg.draw.line(screen, WHITE, (x * cell_size, 0), (x * cell_size, HEIGHT), line_width)
        pg.draw.line(screen, WHITE, (0, x * cell_size), (WIDTH, x * cell_size), line_width)

def draw_markers():
    for row in range(3):
        for col in range(3):
            if board[row][col] == 'X':
                pg.draw.line(screen, RED, (col * cell_size + 25, row * cell_size + 25),
                             ((col + 1) * cell_size - 25, (row + 1) * cell_size - 25), line_width)
                pg.draw.line(screen, RED, ((col + 1) * cell_size - 25, row * cell_size + 25),
                             (col * cell_size + 25, (row + 1) * cell_size - 25), line_width)
            elif board[row][col] == 'O':
                pg.draw.circle(screen, GREEN, (int(col * cell_size + cell_size // 2),
                                               int(row * cell_size + cell_size // 2)), 40, line_width)

def check_winner():
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] is not None:
            return board[row][0]
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
            return board[0][col]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]
    return None
import random

def check_for_win_or_block(char, board):
    # Проверяем, можем ли мы выиграть или заблокировать ход противника
    for row in range(3):
        if board[row].count(char) == 2 and board[row].count(None) == 1:
            return (row, board[row].index(None))
    for col in range(3):
        if [board[row][col] for row in range(3)].count(char) == 2 and [board[row][col] for row in range(3)].count(None) == 1:
            return ([board[row][col] for row in range(3)].index(None), col)
    if [board[i][i] for i in range(3)].count(char) == 2 and [board[i][i] for i in range(3)].count(None) == 1:
        return ([board[i][i] for i in range(3)].index(None), [board[i][i] for i in range(3)].index(None))
    if [board[i][2-i] for i in range(3)].count(char) == 2 and [board[i][2-i] for i in range(3)].count(None) == 1:
        return ([board[i][2-i] for i in range(3)].index(None), 2-[board[i][2-i] for i in range(3)].index(None))
    return None

def minimax(board, depth, isMaximizing):
    winner = check_winner()
    if winner == 'X':
        return -1
    elif winner == 'O':
        return 1
    elif check_draw(board):
        return 0

    if isMaximizing:
        bestScore = -float('inf')
        for row in range(3):
            for col in range(3):
                if board[row][col] is None:
                    board[row][col] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[row][col] = None
                    bestScore = max(score, bestScore)
        return bestScore
    else:
        bestScore = float('inf')
        for row in range(3):
            for col in range(3):
                if board[row][col] is None:
                    board[row][col] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[row][col] = None
                    bestScore = min(score, bestScore)
        return bestScore
    
def ai_turn(board):
    bestScore = -float('inf')
    move = None
    for row in range(3):
        for col in range(3):
            if board[row][col] is None:
                board[row][col] = 'O'
                score = minimax(board, 0, False)
                board[row][col] = None
                if score > bestScore:
                    bestScore = score
                    move = (row, col)
    return move
score_table = {}

def ai_turn_X(board):
    # Заменяем 'X', 'O' и None на числовые значения
    input_board = [[1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row] for row in board]
    # Преобразуем доску в формат, подходящий для модели
    input_board = np.array(input_board)
    input_board = input_board.reshape((1, 3, 3))

    # Получаем предсказания от модели
    predictions = model.predict(input_board)

    # Получаем список доступных клеток
    available_cells = [(i,j) for i in range(3) for j in range(3) if board[i][j] is None]

    # Выбираем клетку с наибольшим предсказанием среди доступных клеток
    available_predictions = [predictions[0][i*3+j] for i,j in available_cells]
    index = np.argmax(available_predictions)
    row, col = available_cells[index]

    return row, col

def ai_turn_O(board):
    # Заменяем 'X', 'O' и None на числовые значения
    input_board = [[1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row] for row in board]
    # Преобразуем доску в формат, подходящий для модели
    input_board = np.array(input_board)
    input_board = input_board.reshape((1, 3, 3))

    # Получаем предсказания от модели
    predictions = model.predict(input_board)

    # Получаем список доступных клеток
    available_cells = [(i,j) for i in range(3) for j in range(3) if board[i][j] is None]

    # Выбираем клетку с наибольшим предсказанием среди доступных клеток
    if len(available_cells) > 0:
        # Используем случайный номер из доступных клеток
        index = np.random.choice(len(available_cells))
        row, col = available_cells[index]
    else:
        row = col = 0

    return row, col

def check_draw(board):
    free_cells = 0
    for row in board:
        for cell in row:
            if cell is None:
                free_cells += 1
    if free_cells == 1:
        running = False
        
    return free_cells == 0

winner = check_winner()

if winner is None and check_draw(board):
    
    running = False

def reset_board():
    global board
    board = [[None, None, None],
             [None, None, None],
             [None, None, None]]
# Игровой цикл
def game_loop():
    global running
    attempt = 1
    running = True
    f = open("log.txt", "a", encoding='utf-8')
    start_time = time.time()
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN and event.key == pg.K_r:
                reset_board()

 
        # AI для 'O'
        if not any(cell is None for row in board for cell in row):
            break
        row, col = ai_turn_O(board)
        board[row][col] = 'O'
        draw_board()
        draw_markers()

        Toe_sound.play()

        winner = check_winner()
        if winner:
            print(f"Победитель: {winner}", file=f)
            print(f"Победитель: {winner}")
            reset_board()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
        elif check_draw(board):
            print("Ничья!", file=f)
            print("Ничья!")
            reset_board()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
        pg.display.update()
        time.sleep(0.5)  # Задержка в 1 секунду

        # AI для 'X'
        row, col = None, None
        while row is None or col is None or board[row][col] is not None:
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()
                    col = (mouse_pos[0] // 100) % 3
                    row = mouse_pos[1] // 100
                    if board[row][col] is not None:
                        row, col = None, None

        board[row][col] = 'X'
        draw_board()
        draw_markers()

        Tic_sound.play()

        winner = check_winner()
        if winner:
            print(f"Победитель: {winner}", file=f)
            print(f"Победитель: {winner}")
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
        elif check_draw(board) and Q == False:
            reset_board()
            print("Ничья!", file=f)
            print("Ничья!")
            reset_board()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
        if winner != None and Q == False:
            reset_board()
            print("Ничья!", file=f)
            print("Ничья!")
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
        pg.display.update()
        time.sleep(0.5)  # Задержка в 1 секунду

        print(f"Попытка: {attempt}", file=f)
        print(f"Попытка: {attempt}")
        attempt += 1
    f.close()



game_loop()
