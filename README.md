# Neurito - Mi cerebro artificial

Edison Javier Gonzalez Velosa
Inteligencia Artificial - Universidad Santo Tomás, Seccional Tunja

Neurito es una mascota virtual controlada por una red neuronal pequeña hecha con NumPy. La red recibe tres entradas (hambre, energía y aburrimiento), tiene una capa oculta de 6 neuronas con ReLU y devuelve tres probabilidades con Softmax: comer, dormir o jugar. La interfaz está hecha con pygame.

## Archivos

- `main.py`: el programa de Neurito. Al inicio tiene las constantes `HIDDEN_UNITS`, `EPOCHS` y `LEARNING_RATE` para repetir los experimentos. Con las teclas 1, 2, 3 y 4 se re-entrena la red con 3, 6, 10 o 20 neuronas ocultas y se muestran la pérdida y la precisión en pantalla.
- `experiments.py`: script que repite los experimentos y imprime los resultados (`python experiments.py`).
- `assets/`: imágenes de Neurito.

## Cómo se entrena

Los ejemplos de entrenamiento se generan con una regla simple: la necesidad más alta decide la acción, y el cansancio se calcula como `1 - energía`. Se usan 3000 ejemplos, 1200 épocas y tasa de aprendizaje 0.1. Con esta configuración la red llega a una pérdida de 0.1225 y una precisión de 99.40 % en 1000 ejemplos de prueba.

## Experimentos

### 1. Dos necesidades fijas y una que varía

La acción cambia cuando la necesidad que varía supera a las demás, igual que en la regla de entrenamiento:

- Con energía 0.5 y aburrimiento 0.3, Neurito pasa de dormir a comer cuando el hambre llega a 0.5.
- Con hambre 0.3 y aburrimiento 0.3, pasa de dormir a comer cuando la energía sube a 0.7 (el cansancio baja a 0.3).
- Con hambre 0.3 y energía 0.5, pasa de dormir a jugar cuando el aburrimiento llega a 0.5.

El cambio no es brusco: la probabilidad de comer sube poco a poco (por ejemplo 2.3 %, 13.3 %, 49.9 %, 86.6 % cuando el hambre va de 0.3 a 0.6).

### 2. Prioridades casi iguales

Cuando dos necesidades están casi empatadas, Softmax reparte la probabilidad casi por igual entre las dos. Ejemplo con hambre 0.6, energía 0.4 y aburrimiento 0.4: comer 50.1 % y dormir 49.0 %. Cuando las tres están empatadas (0.5, 0.5, 0.5) quedan cerca de un tercio cada una (35.2 %, 33.4 % y 31.3 %). En esos casos la red está insegura y la acción elegida casi depende del azar.

### 3. Neuronas en la capa oculta

| Neuronas | Pérdida | Precisión |
|---|---|---|
| 3 | 0.1618 | 96.30 % |
| 6 | 0.1225 | 99.40 % |
| 10 | 0.1262 | 99.30 % |
| 20 | 0.1220 | 98.80 % |

Con 3 neuronas la red es la que peor aprende. De 6 en adelante los resultados son parecidos. Como el resultado cambia un poco con la semilla aleatoria, repetí el experimento con 10 semillas distintas: la precisión promedio fue 78.9 % con 3 neuronas, 97.3 % con 6, 98.7 % con 10 y 99.0 % con 20. Es decir, más neuronas ayudan, pero mejora poco después de 10.

### 4. Épocas y tasa de aprendizaje

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

### 5. Una cuarta acción

Por ejemplo, BAÑARSE. Se necesitaría:

- Una nueva entrada (por ejemplo suciedad), con lo que la red pasa de 3 a 4 entradas.
- Una cuarta salida en la red y un cuarto elemento en la lista de acciones.
- Etiquetas nuevas (0 a 3) generadas con una regla que incluya la nueva necesidad, con ejemplos de las cuatro acciones en cantidades parecidas.
- Imágenes y textos nuevos para esa acción, y volver a entrenar la red.

## Explicación de resultados

### 1. ¿En qué se diferencian pesos y sesgos?

Los pesos multiplican cada entrada y definen cuánta importancia tiene esa entrada para la neurona. El sesgo es un número que se suma al final y mueve el punto en el que la neurona se activa, sin depender de las entradas. En el código, `data @ W1 + b1` hace primero la parte de pesos y luego suma el sesgo.

### 2. ¿Por qué el cansancio es `1 - energía`?

Porque son opuestos: con mucha energía hay poco cansancio y con poca energía hay mucho. Al usar `1 - energía`, las tres necesidades quedan en el mismo sentido (más valor significa más urgencia) y se pueden comparar para ver cuál gana.

### 3. ¿Qué hacen ReLU y Softmax?

ReLU convierte los valores negativos en cero y deja igual los positivos. Se usa en la capa oculta y le da a la red la capacidad de aprender reglas que no son una simple línea recta. Softmax se usa en la salida: convierte los tres resultados en probabilidades que suman 100 %, y la mayor es la acción que Neurito elige.

### 4. ¿Por qué una precisión alta puede ser engañosa?

- Se mide contra una regla que nosotros mismos inventamos, así que 99 % solo dice que la red copió bien esa regla, no que decida bien en el mundo real.
- Depende de la semilla aleatoria: con la misma configuración la precisión fue de 99.4 % con una semilla y de 64.7 % con otra.
- Los pocos errores ocurren en los empates, justo donde la decisión es más difícil, y el número total no lo muestra.
- Si una acción fuera mucho más común que las otras, adivinar siempre esa acción daría buena precisión sin haber aprendido nada.

## Cómo ejecutar


pip install numpy pygame
python main.py
python experiments.py
