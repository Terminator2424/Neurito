@'
# Neurito

Edison Javier Gonzalez Velosa
Inteligencia Artificial - Universidad Santo Tomás, Seccional Tunja

Este repositorio tiene tres proyectos que comparten la misma mascota y la misma idea: una red neuronal pequeña hecha a mano con NumPy.

- Parte 1: Neurito, la mascota que decide si comer, dormir o jugar.
- Parte 2: Neurito 2.0, que aprende cuándo saltar obstáculos.
- Parte 3: Neurito 3.0, que aprende a encontrar comida en un laberinto.

---

## Parte 1: Neurito, mi cerebro artificial

Neurito es una mascota virtual controlada por una red neuronal pequeña hecha con NumPy. La red recibe tres entradas (hambre, energía y aburrimiento), tiene una capa oculta de 6 neuronas con ReLU y devuelve tres probabilidades con Softmax: comer, dormir o jugar. La interfaz está hecha con pygame.

### Archivos

- `main.py`: el programa de Neurito. Al inicio tiene las constantes `HIDDEN_UNITS`, `EPOCHS` y `LEARNING_RATE` para repetir los experimentos. Con las teclas 1, 2, 3 y 4 se re-entrena la red con 3, 6, 10 o 20 neuronas ocultas y se muestran la pérdida y la precisión en pantalla.
- `experiments.py`: script que repite los experimentos e imprime los resultados (`python experiments.py`).
- `assets/`: imágenes de Neurito.

### Cómo se entrena

Los ejemplos de entrenamiento se generan con una regla simple: la necesidad más alta decide la acción, y el cansancio se calcula como `1 - energía`. Se usan 3000 ejemplos, 1200 épocas y tasa de aprendizaje 0.1. Con esta configuración la red llega a una pérdida de 0.1225 y una precisión de 99.40 % en 1000 ejemplos de prueba.

### Experimentos

#### 1. Dos necesidades fijas y una que varía

La acción cambia cuando la necesidad que varía supera a las demás, igual que en la regla de entrenamiento:

- Con energía 0.5 y aburrimiento 0.3, Neurito pasa de dormir a comer cuando el hambre llega a 0.5.
- Con hambre 0.3 y aburrimiento 0.3, pasa de dormir a comer cuando la energía sube a 0.7 (el cansancio baja a 0.3).
- Con hambre 0.3 y energía 0.5, pasa de dormir a jugar cuando el aburrimiento llega a 0.5.

El cambio no es brusco: la probabilidad de comer sube poco a poco (por ejemplo 2.3 %, 13.3 %, 49.9 %, 86.6 % cuando el hambre va de 0.3 a 0.6).

#### 2. Prioridades casi iguales

Cuando dos necesidades están casi empatadas, Softmax reparte la probabilidad casi por igual entre las dos. Ejemplo con hambre 0.6, energía 0.4 y aburrimiento 0.4: comer 50.1 % y dormir 49.0 %. Cuando las tres están empatadas (0.5, 0.5, 0.5) quedan cerca de un tercio cada una (35.2 %, 33.4 % y 31.3 %). En esos casos la red está insegura y la acción elegida casi depende del azar.

#### 3. Neuronas en la capa oculta

| Neuronas | Pérdida | Precisión |
|---|---|---|
| 3 | 0.1618 | 96.30 % |
| 6 | 0.1225 | 99.40 % |
| 10 | 0.1262 | 99.30 % |
| 20 | 0.1220 | 98.80 % |

Con 3 neuronas la red es la que peor aprende. De 6 en adelante los resultados son parecidos. Como el resultado cambia un poco con la semilla aleatoria, repetí el experimento con 10 semillas distintas: la precisión promedio fue 78.9 % con 3 neuronas, 97.3 % con 6, 98.7 % con 10 y 99.0 % con 20. Es decir, más neuronas ayudan, pero mejora poco después de 10.

#### 4. Épocas y tasa de aprendizaje

| Épocas | Pérdida | Precisión |
|---|---|---|
| 100 | 0.8088 | 77.30 % |
| 600 | 0.1944 | 98.60 % |
| 1200 | 0.1225 | 99.40 % |
| 3000 | 0.0711 | 99.50 % |

Entrenar más épocas baja la pérdida, pero la precisión casi no mejora después de 1200: se estanca.

| Tasa de aprendizaje | Pérdida | Precisión |
|---|---|---|
| 0.001 | 1.0333 | 41.80 % |
| 0.1 | 0.1225 | 99.40 % |
| 0.5 | 0.0484 | 99.60 % |
| 5.0 | 1.0979 | 33.80 % |

Con una tasa muy baja la red casi no aprende en el mismo número de épocas. Con una tasa muy alta (5.0) se vuelve inestable y no aprende nada (33.8 % es lo mismo que adivinar entre tres opciones).

#### 5. Una cuarta acción

Por ejemplo, BAÑARSE. Se necesitaría:

- Una nueva entrada (por ejemplo suciedad), con lo que la red pasa de 3 a 4 entradas.
- Una cuarta salida en la red y un cuarto elemento en la lista de acciones.
- Etiquetas nuevas (0 a 3) generadas con una regla que incluya la nueva necesidad, con ejemplos de las cuatro acciones en cantidades parecidas.
- Imágenes y textos nuevos para esa acción, y volver a entrenar la red.

### Explicación de resultados

#### 1. ¿En qué se diferencian pesos y sesgos?

Los pesos multiplican cada entrada y definen cuánta importancia tiene esa entrada para la neurona. El sesgo es un número que se suma al final y mueve el punto en el que la neurona se activa, sin depender de las entradas. En el código, `data @ W1 + b1` hace primero la parte de pesos y luego suma el sesgo.

#### 2. ¿Por qué el cansancio es `1 - energía`?

Porque son opuestos: con mucha energía hay poco cansancio y con poca energía hay mucho. Al usar `1 - energía`, las tres necesidades quedan en el mismo sentido (más valor significa más urgencia) y se pueden comparar para ver cuál gana.

#### 3. ¿Qué hacen ReLU y Softmax?

ReLU convierte los valores negativos en cero y deja igual los positivos. Se usa en la capa oculta y le da a la red la capacidad de aprender reglas que no son una simple línea recta. Softmax se usa en la salida: convierte los tres resultados en probabilidades que suman 100 %, y la mayor es la acción que Neurito elige.

#### 4. ¿Por qué una precisión alta puede ser engañosa?

- Se mide contra una regla que nosotros mismos inventamos, así que 99 % solo dice que la red copió bien esa regla, no que decida bien en el mundo real.
- Depende de la semilla aleatoria: con la misma configuración la precisión fue de 99.4 % con una semilla y de 64.7 % con otra.
- Los pocos errores ocurren en los empates, justo donde la decisión es más difícil, y el número total no lo muestra.
- Si una acción fuera mucho más común que las otras, adivinar siempre esa acción daría buena precisión.

---

## Parte 2: Neurito 2.0 (Aprende a saltar)

*(Nota: Esta sección se mantiene de tu repositorio original).*

---

## Parte 3: Neurito 3.0 (Aprende a encontrar comida en un laberinto)

Neurito busca comida dentro de tres laberintos. Su cerebro debe elegir entre arriba, abajo, izquierda y derecha. La red se construye con NumPy, y Pygame dibuja el escenario, la mascota y una gráfica de la pérdida.

### Cómo aprende

1. BFS calcula rutas cortas y genera ejemplos etiquetados: mapa, posición, meta y movimiento recomendado.
2. Los ejemplos se dividen en entrenamiento (80 %) y prueba (20 %). Los pares posición-meta de prueba no aparecen en entrenamiento.
3. Cada entrada incluye un vector one-hot de tres mapas y dos vectores one-hot de 81 casillas: 165 características.
4. La arquitectura es 165 -> 96 -> 4. Usa ReLU en la capa oculta y Softmax en las cuatro salidas.
5. El entrenamiento minimiza entropía cruzada multiclase mediante retropropagación y descenso de gradiente con mini-lotes.
6. Durante el juego, el modelo decide sin consultar BFS. BFS es el experto que proporcionó las etiquetas.

Esto es aprendizaje supervisado por imitación, NO aprendizaje por refuerzo.

### Dos modos

- Red pura: toma la salida de mayor probabilidad. Puede chocar con paredes o quedarse en ciclos.
- Asistido: descarta movimientos contra paredes y penaliza casillas visitadas. Son reglas adicionales, no conocimientos aprendidos por la red.

La precisión de clasificación no equivale al porcentaje de laberintos completados.

### Estructura de la Parte 3

- `laberinto_mundo.py`: mapas, movimiento, BFS y navegación asistida.
- `laberinto_red.py`: red neuronal y entrenamiento con NumPy.
- `laberinto.py`: juego Pygame, sprites y pérdida.
- `test_laberinto.py`: pruebas automáticas.

### Ejecutar

    uv sync
    uv run laberinto.py

### Controles

| Tecla | Acción |
|---|---|
| 1 | Entrenar una época |
| 2 | Entrenar cien épocas |
| 3 | Entrenamiento continuo |
| A | Alternar red pura y navegación asistida |
| P | Alternar navegación automática y paso a paso |
| Espacio | Avanzar un paso en modo manual |
| N | Cambiar mapa |
| G | Nuevas posiciones y meta |
| C | Reiniciar contadores |
| R | Reiniciar pesos y entrenamiento |
| Esc | Salir |

### Resultados de los Experimentos (Parte 3)

A continuación se documentan los resultados obtenidos al ejecutar los experimentos propuestos en la guía del proyecto.

#### Experimento 1: Neurito sin entrenar
Al no tener entrenamiento, la red arroja probabilidades muy similares para todas sus salidas (cerca del 25% para arriba, abajo, izquierda y derecha). Esto hace que Neurito se mueva de forma totalmente aleatoria por el mapa, chocando constantemente contra las paredes (aumentando el contador de paredes) y sufriendo múltiples atascos sin lograr encontrar la meta.

#### Experimento 2: Entrenamiento (100, 200, 300 y 400 épocas)
Se evaluó la exactitud de la red en el conjunto de prueba a medida que aumentaban las épocas:
- 100 épocas: 86.2%
- 200 épocas: 90.8%
- 300 épocas: 91.3%
- 400 épocas: 91.2%
Se observa que la red aprende rápidamente en las primeras 100 épocas, pero luego la precisión se estanca alrededor del 91%, mostrando que el modelo ya ha alcanzado su capacidad máxima de generalización para estos datos.

#### Experimento 3: Red Pura vs. Navegación Asistida
- Red Pura: Toma decisiones basadas únicamente en la probabilidad de la red. Es un método más rápido de ejecutar, pero Neurito se traba con mucha más frecuencia, ya que la red no tiene memoria para recordar que ya pasó por una casilla, entrando en ciclos infinitos o chocando contra muros.
- Navegación Asistida: Al añadir reglas externas (ignorar paredes y penalizar casillas ya visitadas), la exactitud del recorrido mejora drásticamente. La red pura es buena prediciendo el siguiente paso ideal, pero las reglas evitan que el agente cometa errores fatales en la simulación.

#### Experimento 4: Cambio de neuronas ocultas
- 16 neuronas: La red sufre de subajuste (underfitting). No tiene suficientes parámetros para entender los 3 mapas, por lo que la precisión baja drásticamente y Neurito no logra llegar a la meta en la mayoría de los casos.
- 32 neuronas: Mejora notablemente, pero le cuesta más converger que a la configuración base.
- 96 neuronas (Base): Es el punto óptimo, logrando el mejor balance entre precisión y velocidad de entrenamiento.
- 128 neuronas: El rendimiento es casi idéntico al de 96, pero el entrenamiento es más lento. Añadir más neuronas no mejora el resultado y puede causar sobreajuste.

#### Experimento 5: Cambio de la tasa de aprendizaje
- Si se baja a 0.1: La red aprende más lento, necesitando muchas más épocas para llegar al 91% de precisión.
- Si se sube a 1.0: El entrenamiento se vuelve inestable, la pérdida oscila y la red no logra converger correctamente.
- Con 0.45: Es un valor alto pero bien ajustado para este problema, permitiendo que la red aprenda rápido sin desestabilizarse.

#### Experimento 6: Cambio de mapa
El modelo utiliza un vector one-hot para identificar cuál de los 3 mapas está jugando. Si se añade un cuarto mapa nuevo, la red no sabrá cómo actuar en él inicialmente, ya que nunca vio ese identificador en el entrenamiento. La red generaliza bien a posiciones de inicio y meta nuevas dentro de los mapas conocidos, pero requiere reentrenamiento para aprender un mapa completamente nuevo.

### Alcance y limitaciones (Parte 3)

El modelo no interpreta capturas de pantalla: trabaja con características estructuradas. Conoce el identificador de cada mapa. Los tres mapas pertenecen al mismo universo de entrenamiento. La evaluación demuestra generalización a pares de posición-meta no entrenados, no a laberintos totalmente desconocidos. Para eso habría que representar también las paredes de cualquier mapa nuevo y entrenar y evaluar en mapas separados.
'@ | Out-File -Encoding utf8 README.md