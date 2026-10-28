from pathlib import Path

import numpy as np
import pygame

LANGUAGE = "es"  # en para inglés, es para español

# Parametros para experimentar lo del taller
HIDDEN_UNITS = 6        # Prueba 3, 10 o 20 o usa las teclas 1, 2, 3, 4 en la ventana
EPOCHS = 1200           # Prueba 100, 600, 3000.
LEARNING_RATE = 0.1     # Prueba 0.001, 0.5, 5.0.
SEED = 42

TEXT = {
    "en": {
        "title": "NEURITO - MY ARTIFICIAL BRAIN",
        "hunger": "Hunger", "energy": "Energy", "boredom": "Boredom",
        "eat": "EAT", "sleep": "SLEEP", "play": "PLAY", "neutral": "CALM",
        "accuracy": "Test accuracy against synthetic policy",
        "epoch": "Epoch", "loss": "Loss", "hidden": "Hidden neurons",
        "controls1": "H/J: hunger +/-    E/D: energy +/-",
        "controls2": "A/S: boredom +/-    R: random",
        "controls3": "1/2/3/4: hidden neurons 3/6/10/20",
        "controls4": "N: calm state    ESC: quit",
    },
    "es": {
        "title": "NEURITO - MI CEREBRO ARTIFICIAL",
        "hunger": "Hambre", "energy": "Energía", "boredom": "Aburrimiento",
        "eat": "COMER", "sleep": "DORMIR", "play": "JUGAR", "neutral": "NEUTRAL",
        "accuracy": "Precisión frente a la política sintética",
        "epoch": "Época", "loss": "Pérdida", "hidden": "Neuronas ocultas",
        "controls1": "H/J: hambre +/-    E/D: energía +/-",
        "controls2": "A/S: aburrimiento +/-    R: aleatorio",
        "controls3": "1/2/3/4: neuronas ocultas 3/6/10/20",
        "controls4": "N: estado tranquilo    ESC: salir",
    },
}
labels = TEXT[LANGUAGE]
actions = ("eat", "sleep", "play")


def policy(data):
    """Política sintética: gana la necesidad más alta (cansancio = 1 - energía)."""
    return np.argmax(np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1)


def relu(values):
    return np.maximum(0, values)


def softmax(values):
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def predict(data, params):
    W1, b1, W2, b2 = params
    hidden = relu(data @ W1 + b1)
    return softmax(hidden @ W2 + b2)


def train(hidden_units=HIDDEN_UNITS, epochs=EPOCHS, lr=LEARNING_RATE, seed=SEED):
    """Entrena la red con descenso de gradiente por lote completo."""
    np.random.seed(seed)
    X = np.random.rand(3000, 3)
    targets = policy(X)
    W1 = np.random.randn(3, hidden_units) * 0.5
    b1 = np.zeros((1, hidden_units))
    W2 = np.random.randn(hidden_units, 3) * 0.5
    b2 = np.zeros((1, 3))
    count = len(X)
    loss = float("nan")

    for epoch in range(epochs):
        z1 = X @ W1 + b1
        hidden = relu(z1)
        probabilities = softmax(hidden @ W2 + b2)
        loss = -np.mean(np.log(probabilities[np.arange(count), targets] + 1e-10))
        delta2 = probabilities.copy()
        delta2[np.arange(count), targets] -= 1
        delta2 /= count
        grad_W2 = hidden.T @ delta2
        grad_b2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = (delta2 @ W2.T) * (z1 > 0)
        grad_W1 = X.T @ delta1
        grad_b1 = np.sum(delta1, axis=0, keepdims=True)
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        if epoch % 200 == 0:
            print(f"{labels['epoch']} {epoch} | {labels['loss']}: {loss:.4f}")

    params = (W1, b1, W2, b2)
    test = np.random.rand(1000, 3)
    accuracy = np.mean(np.argmax(predict(test, params), axis=1) == policy(test))
    print(
        f"{labels['hidden']}: {hidden_units} | {labels['loss']}: {loss:.4f} | "
        f"{labels['accuracy']}: {accuracy:.2%}"
    )
    return params, loss, accuracy


params, final_loss, test_accuracy = train()
hidden_units = HIDDEN_UNITS

pygame.init()
screen = pygame.display.set_mode((900, 600))
pygame.display.set_caption("Neurito")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
small_font = pygame.font.SysFont(None, 23)
asset_dir = Path(__file__).resolve().parent / "assets"
pet_size = (330, 330)


def load_frames(names):
    return [
        pygame.transform.smoothscale(
            pygame.image.load(str(asset_dir / name)).convert_alpha(), pet_size
        )
        for name in names
    ]


frames = {
    "neutral": load_frames(["neutral_1.png", "neutral_2.png", "neutral_3.png"]),
    "eat": load_frames(["comer_1.png", "comer_2.png"]),
    "sleep": load_frames(["dormir_1.png", "dormir_2.png"]),
    "play": load_frames(["jugar_1.png", "jugar_2.png"]),
}
needs = [0.2, 0.9, 0.2]
current_state = "neutral"
frame_index = 0
last_frame_time = pygame.time.get_ticks()
running = True
hidden_keys = {pygame.K_1: 3, pygame.K_2: 6, pygame.K_3: 10, pygame.K_4: 20}

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_h:
                needs[0] = min(1, needs[0] + 0.1)
            elif event.key == pygame.K_j:
                needs[0] = max(0, needs[0] - 0.1)
            elif event.key == pygame.K_e:
                needs[1] = min(1, needs[1] + 0.1)
            elif event.key == pygame.K_d:
                needs[1] = max(0, needs[1] - 0.1)
            elif event.key == pygame.K_a:
                needs[2] = min(1, needs[2] + 0.1)
            elif event.key == pygame.K_s:
                needs[2] = max(0, needs[2] - 0.1)
            elif event.key == pygame.K_r:
                needs = list(np.random.rand(3))
            elif event.key == pygame.K_n:
                needs = [0.2, 0.9, 0.2]
            elif event.key in hidden_keys:
                hidden_units = hidden_keys[event.key]
                params, final_loss, test_accuracy = train(hidden_units=hidden_units)

    probabilities = predict(np.array([needs]), params)[0]
    action = actions[int(np.argmax(probabilities))]
    if needs[0] < 0.4 and needs[1] > 0.6 and needs[2] < 0.4:
        state = "neutral"  # Regla de visualización; la red tiene tres salidas.
    else:
        state = action

    now = pygame.time.get_ticks()
    if state != current_state:
        current_state, frame_index, last_frame_time = state, 0, now
    elif now - last_frame_time >= 500:
        frame_index = (frame_index + 1) % len(frames[state])
        last_frame_time = now

    screen.fill((250, 245, 252))
    title = font.render(labels["title"], True, (55, 45, 75))
    screen.blit(title, (45, 28))
    info = small_font.render(
        f"{labels['hidden']}: {hidden_units} | {labels['loss']}: {final_loss:.4f} "
        f"| {test_accuracy:.1%}",
        True, (110, 100, 130),
    )
    screen.blit(info, (47, 62))

    colors = [(235, 125, 75), (70, 175, 115), (105, 125, 225)]
    for index, key in enumerate(("hunger", "energy", "boredom")):
        value = needs[index]
        y = 125 + index * 95
        text = small_font.render(
            f"{labels[key]}: {value * 100:.0f}%", True, (35, 40, 55)
        )
        screen.blit(text, (50, y - 27))
        pygame.draw.rect(screen, (215, 220, 230), (50, y, 290, 22), border_radius=7)
        pygame.draw.rect(
            screen, colors[index], (50, y, int(290 * value), 22), border_radius=7
        )

    pet = frames[state][frame_index]
    screen.blit(pet, pet.get_rect(center=(650, 235)))
    screen.blit(font.render(labels[state], True, (65, 50, 85)), (585, 420))
    for index, action_name in enumerate(actions):
        score = small_font.render(
            f"{labels[action_name]}: {probabilities[index] * 100:.1f}%",
            True, (55, 50, 75)
        )
        screen.blit(score, (50, 380 + index * 28))

    for line, key in enumerate(("controls1", "controls2", "controls3", "controls4")):
        screen.blit(
            small_font.render(labels[key], True, (70, 70, 90)), (50, 480 + line * 27)
        )
    pygame.display.flip()
    clock.tick(30)

pygame.quit()