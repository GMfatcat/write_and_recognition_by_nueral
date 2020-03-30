# press f5 to run
import pygame
import sys, os
import time
import tkinter as tk
from tkinter import messagebox
from pygame.locals import *
pygame.init()
pygame.display.set_caption('Paintme')
mouse = pygame.mouse
fpsClock = pygame.time.Clock()

def message_box(subject, content):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo(subject, content)
    try:
        root.destroy()
    except:
        pass

width = 500
height = 500
window = pygame.display.set_mode((width, height))
canvas = window.copy()
BLACK = pygame.Color( 0 ,  0 ,  0 )
WHITE = pygame.Color(255, 255, 255)
canvas.fill(WHITE)
while True:
    left_pressed, middle_pressed, right_pressed = mouse.get_pressed()
    for event in pygame.event.get():
        if  event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif left_pressed:
            pygame.draw.circle(canvas, BLACK, (pygame.mouse.get_pos()),5)
        elif right_pressed :
            pygame.image.save(window, r"C:\Users\user\Desktop\a.jpg") # fuck yuo !!!!! 記得打位置
            message_box("Good work~", "haha")
            pygame.quit()
            sys.exit()
    window.blit(canvas, (0, 0))
    pygame.draw.circle(window, BLACK, (pygame.mouse.get_pos()), 5)
    pygame.display.update()
