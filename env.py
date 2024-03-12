import pygame
import random
import numpy as np


class SnakeGameAI:
    def __init__(self, width=720, height=480):
        # Initialization
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()

        # Colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)

        # Reset game to initial state
        self.reset()

    def reset(self):
        # Initial snake position
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]

        # Fruit position -- will need to expand on this
        self.fruit_pos = [
            random.randrange(1, (self.width // 10)) * 10,
            random.randrange(1, (self.height // 10)) * 10,
        ]
        self.fruit_spawn = True

        # Game state
        self.direction = "RIGHT"
        self.change_to = self.direction
        self.score = 0

    def step(self, action):
        # Map action to direction -- DRL model will select num from 0-3
        action_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.change_to = action_dict[action]

        # Change direction
        if self.change_to == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        if self.change_to == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        if self.change_to == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        if self.change_to == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

        # Move snake
        if self.direction == "UP":
            self.snake_pos[1] -= 10
        if self.direction == "DOWN":
            self.snake_pos[1] += 10
        if self.direction == "LEFT":
            self.snake_pos[0] -= 10
        if self.direction == "RIGHT":
            self.snake_pos[0] += 10

        # Snake body mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if (
            self.snake_pos[0] == self.fruit_pos[0]
            and self.snake_pos[1] == self.fruit_pos[1]
        ):
            self.score += 10
            self.fruit_spawn = False
        else:
            self.snake_body.pop()

        # Spawn new fruit
        if not self.fruit_spawn:
            self.fruit_pos = [
                random.randrange(1, (self.width // 10)) * 10,
                random.randrange(1, (self.height // 10)) * 10,
            ]
        self.fruit_spawn = True

        # Check game over conditions
        reward = 0
        game_over = False
        if (
            self.snake_pos[0] < 0
            or self.snake_pos[0] > self.width - 10
            or self.snake_pos[1] < 0
            or self.snake_pos[1] > self.height - 10
        ):
            game_over = True
            reward = -10
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                game_over = True
                reward = -10

        # Incremental reward for moving closer to the fruit
        if not game_over:
            reward += 1

        return self.get_game_state(), reward, game_over

    def get_game_state(self):
        # Direction of the snake
        direction_vector = np.zeros(4)
        if self.direction == "UP":
            direction_vector[0] = 1
        elif self.direction == "DOWN":
            direction_vector[1] = 1
        elif self.direction == "LEFT":
            direction_vector[2] = 1
        elif self.direction == "RIGHT":
            direction_vector[3] = 1

        # Relative position of the fruit
        fruit_direction = np.array(self.fruit_pos) - np.array(self.snake_pos)

        # Danger
        danger_straight = 0
        danger_left = 0
        danger_right = 0

        next_position = self.snake_pos.copy()
        if self.direction == "UP":
            next_position[1] -= 10
        elif self.direction == "DOWN":
            next_position[1] += 10
        elif self.direction == "LEFT":
            next_position[0] -= 10
        elif self.direction == "RIGHT":
            next_position[0] += 10

        # Check for immediate danger in front
        if (
            next_position in self.snake_body[1:]
            or next_position[0] < 0
            or next_position[0] > self.width - 10
            or next_position[1] < 0
            or next_position[1] > self.height - 10
        ):
            danger_straight = 1

        # have yet to implement danger_left and danger_right
        state = np.concatenate(
            (
                direction_vector,
                fruit_direction,
                [danger_straight, danger_left, danger_right],
            )
        )

        return state

    def render(self):
        self.display.fill(self.black)
        for pos in self.snake_body:
            pygame.draw.rect(
                self.display, self.green, pygame.Rect(pos[0], pos[1], 10, 10)
            )
        pygame.draw.rect(
            self.display,
            self.red,
            pygame.Rect(self.fruit_pos[0], self.fruit_pos[1], 10, 10),
        )
        pygame.display.flip()
        self.clock.tick(15)

    def show_score(self):
        font = pygame.font.SysFont("arial", 35)
        score_surface = font.render("Score: " + str(self.score), True, self.white)
        score_rect = score_surface.get_rect()
        self.display.blit(score_surface, score_rect)
