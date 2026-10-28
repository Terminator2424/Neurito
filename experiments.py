"""Experimentos para Neurito (sin pygame). Ejecutar: python experiments.py"""
import numpy as np


def policy(data):
    return np.argmax(np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1)


def relu(v):
    return np.maximum(0, v)


def softmax(v):
    e = np.exp(v - np.max(v, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def train(hidden_units=6, epochs=1200, lr=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.rand(3000, 3)
    y = policy(X)
    W1 = np.random.randn(3, hidden_units) * 0.5
    b1 = np.zeros((1, hidden_units))
    W2 = np.random.randn(hidden_units, 3) * 0.5
    b2 = np.zeros((1, 3))
    n = len(X)
    loss = float("nan")
    for _ in range(epochs):
        z1 = X @ W1 + b1
        h = relu(z1)
        p = softmax(h @ W2 + b2)
        loss = -np.mean(np.log(p[np.arange(n), y] + 1e-10))
        d2 = p.copy()
        d2[np.arange(n), y] -= 1
        d2 /= n
        gW2 = h.T @ d2
        gb2 = d2.sum(axis=0, keepdims=True)
        d1 = (d2 @ W2.T) * (z1 > 0)
        gW1 = X.T @ d1
        gb1 = d1.sum(axis=0, keepdims=True)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    test = np.random.rand(1000, 3)
    pred = softmax(relu(test @ W1 + b1) @ W2 + b2)
    acc = np.mean(np.argmax(pred, axis=1) == policy(test))
    net = (W1, b1, W2, b2)
    return loss, acc, net


def predict(net, needs):
    W1, b1, W2, b2 = net
    return softmax(relu(np.array([needs]) @ W1 + b1) @ W2 + b2)[0]


ACTIONS = ("EAT", "SLEEP", "PLAY")


def show(needs, net):
    p = predict(net, needs)
    return ACTIONS[int(np.argmax(p))], p


if __name__ == "__main__":
    loss, acc, net = train()
    print(f"BASE (6 neuronas, 1200 epocas, lr 0.1): loss={loss:.4f} acc={acc:.2%}\n")

    print("EXPERIMENTO 1: variar una necesidad, fijar las otras dos")
    print("-- variar hambre (energia=0.5, aburrimiento=0.3)")
    for v in np.arange(0, 1.01, 0.1):
        a, p = show([v, 0.5, 0.3], net)
        print(f"hambre={v:.1f} -> {a:5s} {np.round(p*100,1)}")
    print("-- variar energia (hambre=0.3, aburrimiento=0.3)")
    for v in np.arange(0, 1.01, 0.1):
        a, p = show([0.3, v, 0.3], net)
        print(f"energia={v:.1f} -> {a:5s} {np.round(p*100,1)}")
    print("-- variar aburrimiento (hambre=0.3, energia=0.5)")
    for v in np.arange(0, 1.01, 0.1):
        a, p = show([0.3, 0.5, v], net)
        print(f"aburr={v:.1f} -> {a:5s} {np.round(p*100,1)}")

    print("\nEXPERIMENTO 2: prioridades casi empatadas")
    for needs in ([0.50, 0.50, 0.50], [0.60, 0.40, 0.40], [0.61, 0.39, 0.60],
                  [0.70, 0.30, 0.69], [0.90, 0.10, 0.10], [0.20, 0.90, 0.20]):
        a, p = show(needs, net)
        print(f"{needs} -> {a:5s} {np.round(p*100,1)}")

    print("\nEXPERIMENTO 3: neuronas ocultas")
    for h in (3, 6, 10, 20):
        l, ac, _ = train(hidden_units=h)
        print(f"hidden={h:2d} loss={l:.4f} acc={ac:.2%}")

    print("\nEXPERIMENTO 4a: epocas (lr=0.1)")
    for e in (100, 300, 600, 1200, 3000):
        l, ac, _ = train(epochs=e)
        print(f"epochs={e:4d} loss={l:.4f} acc={ac:.2%}")

    print("\nEXPERIMENTO 4b: tasa de aprendizaje (1200 epocas)")
    for lr in (0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 20.0):
        l, ac, _ = train(lr=lr)
        print(f"lr={lr:<6} loss={l:.4f} acc={ac:.2%}")

    print("\nEXPERIMENTO 5: robustez a la semilla (config base)")
    for s in (0, 1, 7, 42, 123):
        l, ac, _ = train(seed=s)
        print(f"seed={s:3d} loss={l:.4f} acc={ac:.2%}")

    print("\nACIERTO ENGANOSO: clase mayoritaria")
    np.random.seed(42)
    T = np.random.rand(100000, 3)
    c = np.bincount(policy(T)) / len(T)
    print("proporcion por clase", np.round(c, 3), "-> siempre predecir la mayor:", f"{c.max():.2%}")
    # error cerca de la frontera
    _, _, net2 = train()
    W1, b1, W2, b2 = net2
    pr = np.argmax(softmax(relu(T @ W1 + b1) @ W2 + b2), axis=1)
    sc = np.column_stack((T[:, 0], 1 - T[:, 1], T[:, 2]))
    srt = np.sort(sc, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    err = pr != policy(T)
    print(f"errores totales: {err.mean():.2%}; margen medio de los errores: {margin[err].mean():.3f} vs margen medio global: {margin.mean():.3f}")
    print(f"errores con margen < 0.05: {np.mean(err[margin<0.05]):.2%} ; con margen >= 0.05: {np.mean(err[margin>=0.05]):.2%}")