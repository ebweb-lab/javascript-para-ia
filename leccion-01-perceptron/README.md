# Lección 1 — Una neurona desde cero: un perceptrón en JS puro

> Curso **JavaScript para IA**. Construimos la unidad más básica de una red neuronal, el perceptrón, sin librerías, y la vemos aprender en directo a clasificar puntos en un `<canvas>`.

---

## Índice

1. [Objetivos](#objetivos)
2. [Requisitos](#requisitos)
3. [Cómo ejecutarla](#cómo-ejecutarla)
4. [Estructura de archivos](#estructura-de-archivos)
5. [Conceptos teóricos](#conceptos-teóricos)
6. [Explicación del código](#explicación-del-código)
   - [perceptron.js](#perceptronjs--el-cerebro)
   - [demo.js](#demojs--la-visualización)
   - [index.html](#indexhtml--la-página)
7. [Ejercicios](#ejercicios)

---

## Objetivos

Al terminar la lección el alumno sabrá:

- Qué es una neurona artificial: **entradas, pesos, sesgo, suma ponderada y activación**.
- Cómo aprende un perceptrón con la **regla de Rosenblatt** (1958).
- Qué son la **tasa de aprendizaje** y la **época**.
- Por qué un perceptrón dibuja una **frontera de decisión recta**.
- Cuál es su gran limitación: solo resuelve problemas **linealmente separables** (no puede con XOR).
- Cómo visualizar un modelo de ML con la API de Canvas 2D.

## Requisitos

- JavaScript básico: clases, arrays, bucles, funciones flecha.
- Un navegador moderno (Chrome, Edge, Firefox).
- Node es opcional: solo para una prueba rápida en la terminal. Sin npm ni servidor.

## Cómo ejecutarla

Haz doble clic en `index.html`. Los scripts son clásicos (no son módulos ES), así que funciona abriéndolo directamente con `file://`.

Para probar solo la neurona en Node:

```bash
node -e "eval(require('fs').readFileSync('perceptron.js','utf8')+';globalThis.P=Perceptron'); const n=new P(2); console.log(n.predecir([0.5,0.5]))"
```

## Estructura de archivos

```
leccion-01-perceptron/
├── index.html      → teoría, código explicado, demo interactiva y ejercicios
├── perceptron.js   → la clase Perceptron (51 líneas). El núcleo de la lección
├── demo.js         → datos, bucle de entrenamiento, dibujo en canvas y eventos
└── README.md       → este archivo
```

El orden de carga importa. `index.html` carga primero `perceptron.js` y después `demo.js` (líneas 184–185), porque la demo usa la clase `Perceptron`.

---

## Conceptos teóricos

### La neurona

Una neurona recibe números (entradas), multiplica cada uno por su **peso**, suma un **sesgo** y aplica una **función de activación**:

```
z = w₁·x + w₂·y + b
salida = z ≥ 0 ? +1 : −1        (activación escalón)
```

- **Pesos (`w`)** indican cuánto importa cada entrada y en qué dirección.
- **Sesgo (`b`)** desplaza la decisión. Sin él, la frontera estaría obligada a pasar por el origen (0,0).
- **Activación** convierte el número `z` en una decisión.

### La frontera de decisión

Los puntos donde la neurona "duda" cumplen `w₁·x + w₂·y + b = 0`, y eso es la ecuación de una **recta**. Despejando `y`:

```
y = −(w₁·x + b) / w₂
```

Todo lo que queda a un lado es clase A (+1) y al otro clase B (−1). **Aprender = mover esa recta.**

### La regla de aprendizaje

Para cada ejemplo `(entradas, objetivo)`:

```
error = objetivo − predicción          ∈ { −2, 0, 2 }
wᵢ   += tasa · error · entradaᵢ
b    += tasa · error
```

- Si acierta, `error = 0` y no cambia nada.
- Si falla, los pesos se mueven hacia la respuesta correcta, en proporción a cada entrada.
- La **tasa de aprendizaje** controla el tamaño del paso: alta significa rápido pero a saltos; baja significa suave pero lento.
- Una **época** es una pasada completa por todos los datos.

### Teorema de convergencia

Si los datos **se pueden separar con una recta**, el perceptrón **siempre** termina encontrando una que los separa, en un número finito de pasos. Si no se pueden separar (ejemplo: XOR, o un círculo de A rodeado de B), nunca converge. Esta limitación motiva la Lección 2: redes multicapa.

---

## Explicación del código

### `perceptron.js` — el cerebro

| Líneas | Qué hace |
|-------:|----------|
| 6 | `class Perceptron`: toda la neurona en una clase. |
| 11–16 | **Constructor.** Crea `numEntradas` pesos aleatorios en [−1, 1] (l.13), un sesgo aleatorio (l.14) y guarda la tasa (l.15). La neurona nace sin saber nada. |
| 19–25 | **`sumaPonderada`.** Empieza en el sesgo (l.20) y acumula `peso × entrada` (l.22). Devuelve `z`. |
| 28–30 | **`activacion`.** Función escalón: `z >= 0 ? 1 : -1`. |
| 33–35 | **`predecir`.** Encadena suma y activación. |
| 41–50 | **`entrenar`.** Calcula el error (l.42). Si falla (l.43), actualiza cada peso (l.45) y el sesgo (l.47). Devuelve el error para que quien llama pueda contar fallos. |

### `demo.js` — la visualización

**Configuración y estado**

| Líneas | Qué hace |
|-------:|----------|
| 6–7 | Obtiene el canvas y su contexto 2D. |
| 9–25 | Objeto `ui` con referencias a todos los elementos del HTML. |
| 28–35 | Estado global: la neurona, los puntos, la recta secreta, la época, el índice del punto actual, los errores de la época, el modo auto y el último punto entrenado. |

**Datos**

| Líneas | Qué hace |
|-------:|----------|
| 38–40 | `nuevaLineaSecreta`: recta aleatoria `y = m·x + c`. Es la "verdad" que la neurona no conoce. |
| 41–43 | `etiquetaReal`: +1 si el punto está por encima de la recta secreta y −1 si está por debajo. Así se generan las etiquetas, igual que un dataset etiquetado. |
| 45–53 | `generarPuntos`: crea 80 puntos aleatorios con su etiqueta. |
| 55–64 | `reiniciarEntrenamiento`: neurona nueva y contadores a cero. |
| 66–71 | `barajar`: Fisher-Yates. Barajar cada época evita que el orden de los datos sesgue el aprendizaje. |

**Entrenamiento**

| Líneas | Qué hace |
|-------:|----------|
| 75–101 | `paso`: entrena con **un** punto (l.78), cuenta el error y avanza. Al terminar la época (l.84) comprueba si hubo cero errores, en cuyo caso **convergió** (l.90), o si se pasó de 500 épocas, en cuyo caso los datos no son separables (l.95). |
| 103–111 | `bucle`: en modo auto ejecuta N pasos por frame (control "velocidad") y repinta con `requestAnimationFrame` (l.110). |
| 113–116 | `detenerAuto`: pausa. |

**Dibujo**

| Líneas | Qué hace |
|-------:|----------|
| 119–121 | `css()`: lee los colores de las variables CSS, así el canvas respeta el modo claro u oscuro. |
| 124–131 | `ajustarTamano`: escala el canvas según `devicePixelRatio` para que se vea nítido en pantallas HiDPI. |
| 134–135 | `aPx` y `aMundo`: convierten entre coordenadas del mundo [−1, 1] y píxeles. La `y` se invierte porque en el canvas crece hacia abajo. |
| 137–211 | `dibujar`, por capas: |
| 141–149 | · **Fondo**: una rejilla de 40×40 donde cada celda se pinta según lo que predice la neurona. Muestra visualmente su "opinión" sobre todo el plano. |
| 151–157 | · Ejes. |
| 159–166 | · Recta secreta (punteada). |
| 168–174 | · **Frontera aprendida** con `y = −(w₁·x + b) / w₂`. Se protege contra `w₂ ≈ 0` (l.170). |
| 176–194 | · Puntos coloreados según su clase, con un anillo rojo si están mal clasificados (l.180 compara predicción y etiqueta). |
| 196–204 | · Anillo en el último punto entrenado. |
| 206–210 | · Panel: época, precisión, w₁, w₂ y sesgo. |
| 214–220 | `linea`: dibuja un segmento entre dos puntos del mundo. |

**Eventos**

| Líneas | Qué hace |
|-------:|----------|
| 224–225 | `leerTasa`: el slider es logarítmico (−3 a 0) y la tasa es `10^valor`, de 0.001 a 1. Así los valores pequeños ocupan tanto recorrido como los grandes. |
| 228–236 | Botones: un paso, entrenar o pausar, reiniciar, nuevos puntos, vaciar. |
| 237–240 | El slider de tasa cambia `neurona.tasa` en caliente y muestra el valor real con 2 cifras significativas. |
| 244–251 | Clic izquierdo añade un punto A, clic derecho o Shift+clic añade un punto B. Sirve para crear datasets propios, incluidos los no separables. |
| 253–254 | Repinta al cambiar el tema o el tamaño de la ventana. |
| 256 | Arranque: genera los puntos iniciales. |

### `index.html` — la página

| Líneas | Sección |
|-------:|---------|
| 7–75 | Estilos. Colores como variables CSS en `:root`, con versión oscura dentro de `prefers-color-scheme`. |
| 84 | §1 La idea (fórmula). |
| 89 | §2 Cómo aprende (regla). |
| 98 | §3 El código (clase completa y ejemplo de uso). |
| 134 | §4 Pruébalo: demo, con el canvas en la l.138. |
| 165 | §5 Qué observar. |
| 172 | §6 Ejercicios. |
| 184–185 | Carga de `perceptron.js` y `demo.js`. |

---

## Ejercicios

1. **Rompe la neurona.** Vacía el lienzo, dibuja un grupo A en el centro y rodéalo de puntos B. ¿Por qué no converge? (Investiga *XOR* y *separabilidad lineal*.)
2. **Sin sesgo.** En `perceptron.js`, l.14, pon `this.sesgo = 0` y comenta la l.47. ¿Qué le pasa a la recta?
3. **Sigmoide.** Añade `sigmoide(z) { return 1 / (1 + Math.exp(-z)); }` y muestra en consola la "confianza" de cada predicción.
4. **Puertas lógicas.** Con `new Perceptron(2)` entrena AND, después OR, después XOR, con entradas 0/1 y objetivos −1/+1.
5. **Mide.** ¿Cuántas épocas tarda en converger con 20, 80 y 300 puntos? Para cambiar el número de puntos, edita el valor por defecto de `generarPuntos(n = 80)` en `demo.js`, l.45. Haz una tabla con los resultados.

<details>
<summary>Soluciones (pistas)</summary>

1. Ninguna recta puede separar un círculo de su anillo exterior, y el perceptrón solo sabe dibujar rectas.
2. Sin sesgo, la frontera `w₁·x + w₂·y = 0` pasa siempre por el origen (0,0).
3. La sigmoide da un valor entre 0 y 1: cerca de 0.5 significa duda, cerca de 0 o 1 significa seguridad.
4. AND y OR convergen. XOR nunca converge.
5. Más puntos normalmente implica más épocas, sobre todo si hay puntos muy cerca de la recta secreta.

</details>
