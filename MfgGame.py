import numpy as np
import pygame
import sys


# data
demand = 200
duration = 100
num_cfgs = 5
cfgs_limit = 10
purchase_costs = np.array([[1000.0, 1500.0, 2000.0, 500.0, 5000.0]]).T
running_costs = np.array([[10.0, 15.0, 20.0, 5.0, 25.0]]).T
production_rates = np.array([[1, 1.5, 2, 0.25, 5]]).T
setup_times = np.array([[5, 7.5, 9, 3.5, 10]]).T

# PG
pygame.init()
screen = pygame.display.set_mode((1280, 768), pygame.SCALED)
pygame.display.set_caption("Manufacturing Environment")
background = pygame.Surface(screen.get_size()).convert()
background.fill((32, 42, 68))

if pygame.font:
    # render main title
    text = pygame.font.Font(None, 32).render(
        "ManufacturingRL Environment", True, (255, 255, 255)
    )
    textpos = text.get_rect(centerx=background.get_width() / 2, y=10)
    background.blit(text, textpos)

    # render available configurations title
    text = pygame.font.Font(None, 28).render(
        "Available configurations", True, (255, 255, 255)
    )
    textpos = text.get_rect(x=50, y=75)
    background.blit(text, textpos)

    # render configurations properties
    font = pygame.font.Font(None, 24)
    txt_x, txt_y = 60, 100
    line_spacing = 25
    for i in range(num_cfgs):
        text = font.render(f"{'':>5} • Mfg-{i}:", True, (255, 255, 255))
        textpos = text.get_rect(x=txt_x, y=txt_y)
        background.blit(text, textpos)
        txt_y += line_spacing

        text = font.render(
            f"{'':>10} - Production rate: {production_rates[i][0]}",
            True,
            (255, 255, 255),
        )
        textpos = text.get_rect(x=txt_x, y=txt_y)
        background.blit(text, textpos)
        txt_y += line_spacing

        text = font.render(
            f"{'':>10} - Purchase cost: {purchase_costs[i][0]}", True, (255, 255, 255)
        )
        textpos = text.get_rect(x=txt_x, y=txt_y)
        background.blit(text, textpos)
        txt_y += line_spacing

        text = font.render(
            f"{'':>10} - Running cost: {running_costs[i][0]}", True, (255, 255, 255)
        )
        textpos = text.get_rect(x=txt_x, y=txt_y)
        background.blit(text, textpos)
        txt_y += line_spacing


screen.blit(background, (0, 0))
pygame.display.flip()

clock = pygame.time.Clock()

going = True
while going:
    clock.tick(60)

    # Handle Input Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            going = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            going = False

    # Draw Everything
    screen.blit(background, (0, 0))
    pygame.display.flip()

pygame.quit()
