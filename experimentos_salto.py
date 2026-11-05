"""Experimentos 3 y 4 sin ventana: mismos datos y misma pista para todos."""

from salto_juego import JuegoSalto
from salto_modelo import RedSalto

EPOCAS = 300
FRAMES = 1800  # 60 segundos a 30 fps


def jugar(red):
    juego = JuegoSalto()
    for _ in range(FRAMES):
        if red.probabilidad(juego.distancia, juego.velocidad) >= 0.5 and juego.en_suelo:
            juego.saltar()
        juego.actualizar()
    return juego.esquivados, juego.choques


def correr(nombre, **parametros):
    red = RedSalto(**parametros)
    X, y = red.generar_datos(5000)
    X_test, y_test = red.generar_datos(1500)
    for _ in range(EPOCAS):
        red.entrenar_epoca(X, y)
    esquivados, choques = jugar(red)
    print(f"| {nombre} | {red.historial[-1]:.4f} | "
          f"{red.precision(X_test, y_test):.1%} | {esquivados} | {choques} |")


print("## Experimento 3: neuronas ocultas (tasa 0.8, 300 épocas)")
print("| Ocultas | Pérdida | Precisión | Esquivados | Choques |")
print("|---|---|---|---|---|")
for n in (3, 12, 24):
    correr(n, ocultas=n)

print()
print("## Experimento 4: tasa de aprendizaje (12 ocultas, 300 épocas)")
print("| Tasa | Pérdida | Precisión | Esquivados | Choques |")
print("|---|---|---|---|---|")
for t in (0.01, 0.8, 1.2):
    correr(t, tasa=t)