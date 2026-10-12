# Lección 2 — Varias neuronas en capas: una red que aprende XOR

> Curso **JavaScript para IA**. En la lección anterior construimos una neurona que solo sabe trazar rectas. Hoy juntamos varias en una red de dos capas, la entrenamos con **descenso del gradiente** y **retropropagación**, y la vemos resolver problemas que una sola neurona nunca podría: XOR, un círculo y una espiral.

---

## Índice

1. [Qué vas a aprender](#qué-vas-a-aprender)
2. [Antes de empezar](#antes-de-empezar)
3. [El muro: XOR](#el-muro-xor)
4. [La idea: poner neuronas en capas](#la-idea-poner-neuronas-en-capas)
5. [Construyendo la red paso a paso](#construyendo-la-red-paso-a-paso)
6. [Cómo aprende: pérdida y descenso del gradiente](#cómo-aprende-pérdida-y-descenso-del-gradiente)
7. [Retropropagación en el código](#retropropagación-en-el-código)
8. [La prueba: XOR resuelto](#la-prueba-xor-resuelto)
9. [La demo por dentro](#la-demo-por-dentro)
10. [Experimenta](#experimenta)
11. [Resultados medidos](#resultados-medidos)
12. [Ejercicios](#ejercicios)

---

## Qué vas a aprender

Al terminar esta lección vas a entender:

- Por qué un perceptrón no puede resolver **XOR**.
- Qué es una **capa oculta** y cómo varias rectas combinadas forman fronteras curvas.
- Por qué hacen falta **activaciones suaves** como tanh y sigmoide.
- Qué mide una **función de pérdida** y cómo se usa la pérdida logarítmica.
- Cómo funciona el **descenso del gradiente**.
- Cómo la **retropropagación** reparte el error entre los pesos usando la regla de la cadena.
- Qué son los **mínimos locales** y por qué la arquitectura y la tasa de aprendizaje importan tanto.

Todo en JavaScript puro, en unas 80 líneas, sin librerías.

## Antes de empezar

Necesitas haber hecho la [Lección 1](../leccion-01-perceptron/), porque partimos del perceptrón. En JavaScript basta con manejar clases, `map`, arrays anidados y desestructuración. En matemáticas, con saber qué es una pendiente: la derivada la vamos a usar de forma intuitiva.

Para abrir la lección, haz doble clic en `index.html`. Los scripts son clásicos (no módulos ES), así que funciona directamente desde el disco, sin servidor ni `npm install`.

Los archivos son pocos:

```
leccion-02-red-multicapa/
├── index.html   → la lección en el navegador: teoría, diagrama, código y demo
├── red.js       → la red neuronal. Es el corazón de la lección
├── demo.js      → los datos de ejemplo, el entrenamiento en vivo y los dibujos
└── README.md    → este archivo
```

`index.html` carga primero `red.js` y después `demo.js`, porque la demo usa la clase `RedNeuronal`.

---

## El muro: XOR

XOR significa «uno u otro, pero no los dos»:

| a | b | a XOR b |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Si pones estos cuatro puntos en un plano, los unos quedan en una diagonal y los ceros en la otra. No hay ninguna recta que los separe, y una recta es lo único que sabe dibujar el perceptrón de la Lección 1. Puedes comprobarlo: vacía su lienzo, dibuja estos cuatro puntos y dale a entrenar. Nunca converge.

Minsky y Papert señalaron esta limitación en 1969, y contribuyó a frenar la investigación en redes neuronales durante años. La salida, sin embargo, es sencilla: con **dos** rectas sí se puede encerrar una diagonal. Solo falta una forma de combinarlas.

## La idea: poner neuronas en capas

```
entradas      capa oculta            salida
  x  ─┐      ┌─ h₁ = tanh(W1₁·[x,y] + b1₁) ─┐
      ├──────┼─ h₂ = tanh(W1₂·[x,y] + b1₂) ─┼──  p = sigmoide(W2·h + b2)
  y  ─┘      └─ h₃ = tanh(W1₃·[x,y] + b1₃) ─┘
```

Ponemos varias neuronas una al lado de la otra: eso es la **capa oculta**. Cada una es un perceptrón como el de la lección anterior y traza su propia recta. Después añadimos una neurona de salida que no mira el punto, sino las respuestas de las ocultas, y las combina. Con varias rectas combinadas aparecen franjas, triángulos, polígonos y, en la práctica, casi cualquier forma.

Hay un segundo cambio. La salida ya no es +1 o −1, sino un número entre 0 y 1 que leemos como la **probabilidad de que el punto sea de clase A**. Un 0.9 significa «bastante seguro de que es A». Por eso, en esta lección, las etiquetas pasan a ser `1` y `0`.

---

## Construyendo la red paso a paso

### Cuatro herramientas

`red.js` empieza con cuatro funciones pequeñas. La primera genera un número aleatorio entre −1 y 1, para los pesos iniciales:

```js
const aleatorio = () => Math.random() * 2 - 1;
```

La segunda es la **sigmoide**: toma cualquier número y lo comprime entre 0 y 1. Los muy negativos dan casi 0, los muy positivos casi 1 y el cero da exactamente 0.5. Es la que convierte la salida en una probabilidad.

```js
const sigmoide = (z) => 1 / (1 + Math.exp(-z));
```

La tercera es el **producto escalar**: multiplicar dos listas elemento a elemento y sumar. Es la suma ponderada de la Lección 1, ahora como función reutilizable, porque la vamos a necesitar en las dos capas.

```js
function productoPunto(a, b) {
  let suma = 0;
  for (let i = 0; i < a.length; i++) suma += a[i] * b[i];
  return suma;
}
```

La cuarta es la **pérdida**, que veremos enseguida.

### El constructor: matrices de pesos

La capa oculta necesita una fila de pesos por neurona, y cada fila tiene un peso por entrada. Eso es una **matriz**, que en JavaScript es un array de arrays. Con 2 entradas y 3 ocultas, `W1` es de 3 × 2, y cada fila es exactamente un perceptrón de la lección anterior. `b1` guarda un sesgo por neurona oculta.

```js
constructor(numEntradas, numOcultas, tasa = 0.1) {
  this.tasa = tasa;
  this.W1 = Array.from({ length: numOcultas }, () =>
    Array.from({ length: numEntradas }, aleatorio));
  this.b1 = Array.from({ length: numOcultas }, aleatorio);
  this.W2 = Array.from({ length: numOcultas }, aleatorio);
  this.b2 = aleatorio();
}
```

La salida es una sola neurona: tiene un peso por cada neurona oculta (`W2`) y su propio sesgo (`b2`).

### Por qué el escalón ya no sirve

En la Lección 1 la activación era un escalón: +1 o −1. Aquí no podemos usarlo. Para aprender, la red necesita saber cuánto cambia la salida si mueve un poco un peso, es decir, la **derivada**. El escalón es plano en todas partes salvo en el salto, así que su derivada es 0 y no aporta ninguna información.

Por eso usamos activaciones suaves:

| Función | Rango | Derivada (en función de su salida) | Dónde la usamos |
|---|---|---|---|
| `tanh(z)` | (−1, 1) | `1 − h²` | capa oculta |
| `sigmoide(z)` | (0, 1) | `p·(1 − p)` | salida |

tanh es, en el fondo, un escalón con las esquinas redondeadas.

### Propagación hacia delante

Con todo esto, calcular la respuesta de la red es directo:

```js
propagar(entradas) {
  const ocultas = this.W1.map((pesos, j) =>
    Math.tanh(productoPunto(pesos, entradas) + this.b1[j]));
  const salida = sigmoide(productoPunto(this.W2, ocultas) + this.b2);
  return { ocultas, salida };
}
```

Cada neurona oculta hace su suma ponderada, le añade su sesgo y la pasa por tanh. La salida hace lo mismo con las ocultas y la pasa por sigmoide. Devolvemos también `ocultas` porque el entrenamiento las necesita. `predecir` simplemente devuelve `propagar(entradas).salida`.

---

## Cómo aprende: pérdida y descenso del gradiente

### Medir cuánto se equivoca

En la Lección 1 el error era de todo o nada. Ahora queremos saber **cuánto** se equivoca la red, y para eso usamos la pérdida logarítmica (entropía cruzada binaria):

```
L = −[ t·log(p) + (1 − t)·log(1 − p) ]
```

Si el objetivo es 1 y la red dice 0.99, la pérdida es casi 0. Si dice 0.01, es decir, si se equivoca con mucha seguridad, la pérdida se dispara. Castiga sobre todo los errores seguros.

```js
function perdida(p, t) {
  const e = 1e-12; // evita log(0)
  return -(t * Math.log(p + e) + (1 - t) * Math.log(1 - p + e));
}
```

El `1e-12` impide calcular el logaritmo de 0, que es menos infinito y rompería los cálculos.

### Bajar la montaña

Entrenar es minimizar `L`. Imagina la pérdida como una montaña donde tu posición son los pesos. El **gradiente** `∂L/∂w` dice cuánto sube la pérdida si sube un peso; si lo restamos, bajamos:

```
w ← w − tasa · ∂L/∂w
```

La **tasa de aprendizaje** es el tamaño del paso. Si es demasiado grande, saltas de un lado al otro del valle. Si es demasiado pequeña, tardas muchísimo en llegar.

La superficie de la pérdida tiene varios valles, y según los pesos iniciales la red puede quedarse en uno que no es el más profundo. Eso es un **mínimo local**. En la práctica se resuelve reiniciando con otros pesos, añadiendo neuronas o ajustando la tasa.

---

## Retropropagación en el código

Este es el centro de la lección. `entrenar` aprende de un solo ejemplo en tres pasos. Primero propaga hacia delante y guarda las activaciones:

```js
const { ocultas, salida } = this.propagar(entradas);
```

**Paso 1: el error en la salida.** Al derivar la pérdida logarítmica combinada con la sigmoide, casi todo se cancela y queda algo muy simple: salida menos objetivo. Por eso estas dos funciones suelen ir juntas.

```js
const deltaSalida = salida - objetivo;
```

**Paso 2: repartir la culpa hacia atrás.** La responsabilidad de cada neurona oculta es el producto de tres factores: cuánto se equivocó la salida, cuánto pesaba esa neurona en la salida y la pendiente de tanh en ese punto (`1 − h²`). Multiplicar factores al ir hacia atrás es la **regla de la cadena**, y eso es literalmente la retropropagación.

```js
const deltaOcultas = ocultas.map((h, j) =>
  deltaSalida * this.W2[j] * (1 - h * h));
```

Un detalle que causa muchos errores: esto se calcula **antes** de modificar `W2`, porque hay que usar los pesos que produjeron el error, no los ya corregidos.

**Paso 3: el descenso del gradiente.** El gradiente de cualquier peso tiene siempre la misma forma: la culpa (δ) de la neurona a la que llega, multiplicada por el valor que transporta esa conexión. Los sesgos funcionan igual, con una entrada que siempre vale 1.

```js
for (let j = 0; j < this.W2.length; j++) {
  this.W2[j] -= this.tasa * deltaSalida * ocultas[j];
}
this.b2 -= this.tasa * deltaSalida;

for (let j = 0; j < this.W1.length; j++) {
  for (let i = 0; i < entradas.length; i++) {
    this.W1[j][i] -= this.tasa * deltaOcultas[j] * entradas[i];
  }
  this.b1[j] -= this.tasa * deltaOcultas[j];
}

return perdida(salida, objetivo);
```

Al final devuelve la pérdida del ejemplo, que la demo usa para dibujar la gráfica.

Compáralo con la regla de la Lección 1: `w += tasa · error · entrada`. Es la misma forma. El perceptrón ya hacía un descenso del gradiente rudimentario sin saberlo; aquí el error es continuo y se reparte entre capas. El signo cambia porque aquí δ es `salida − objetivo`, mientras que allí el error era `objetivo − predicción`.

---

## La prueba: XOR resuelto

Desde la carpeta de la lección puedes entrenar la red con XOR en Node: 3 neuronas ocultas, tasa 0.5 y 3000 épocas.

```bash
node -e "eval(require('fs').readFileSync('red.js','utf8')+';globalThis.R=RedNeuronal'); const n=new R(2,3,0.5); const d=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]; for(let e=0;e<3000;e++) for(const[a,b,t] of d) n.entrenar([a,b],t); d.forEach(([a,b])=>console.log(a,b,n.predecir([a,b]).toFixed(3)))"
```

Una ejecución típica:

```
0 0 0.000
0 1 0.999
1 0 0.999
1 1 0.001
```

Los pesos iniciales son aleatorios, así que alguna vez la red puede quedarse en un mínimo local. Si pasa, vuelve a ejecutarlo.

---

## La demo por dentro

`demo.js` no forma parte de la red; es lo que te permite verla aprender. Lo importante:

- **Tres conjuntos de datos.** *XOR por cuadrantes* (los cuadrantes 2 y 4 son clase A, con puntos alejados de los ejes para que la frontera se vea clara), *círculo* (clase A dentro de un radio de 0.6) y *espiral* (dos brazos enrollados, con algo de ruido). XOR y círculo usan 160 puntos; la espiral, 240.
- **Una época** es pasar una vez por todos los puntos, barajados para que el orden no sesgue el aprendizaje. Se guarda la pérdida media de cada época para la gráfica:

  ```js
  function entrenarEpoca() {
    if (puntos.length === 0) return;
    barajar(puntos);
    let suma = 0;
    for (const p of puntos) suma += red.entrenar([p.x, p.y], p.etiqueta);
    historial.push(suma / puntos.length);
    epoca++;
  }
  ```

- **El entrenamiento automático** ejecuta varias épocas por fotograma (el control «Velocidad») y se detiene solo cuando la pérdida baja de 0.01.
- **El fondo es un mapa de probabilidad.** Se pregunta a la red en una rejilla de 50 × 50 y cada celda se pinta con el color de la clase A usando `globalAlpha = p`. Así se ve dónde está segura la red y dónde duda.
- **Las rectas punteadas** son las de cada neurona oculta: `W1[j][0]·x + W1[j][1]·y + b1[j] = 0`, la misma frontera de la Lección 1, una por neurona.
- **Los controles.** Cambiar el número de neuronas ocultas reinicia la red, porque cambia su forma. La tasa se puede cambiar en caliente: el slider es logarítmico y va de 0.001 a 1. Con clic izquierdo añades puntos de clase A y con clic derecho (o Shift + clic), de clase B.

---

## Experimenta

Abre `index.html` y prueba esto en orden:

1. **XOR con 1 neurona oculta.** Es un perceptrón disfrazado: una sola recta. No puede.
2. **XOR con 2 neuronas.** Mejora, pero se queda en torno al 85 %.
3. **XOR con 4 neuronas.** Las rectas se colocan sobre los ejes y lo resuelve.
4. **Mínimos locales.** Vuelve a 2 neuronas y pulsa varias veces *Reiniciar red* y *Entrenar*. Cada intento termina distinto: a veces la red se atasca en un valle que no es el mejor.
5. **Círculo con 2, 3 y 8 neuronas.** Dos rectas no encierran nada, tres forman un triángulo y ocho casi un círculo. Fíjate en la curva de pérdida: mesetas largas y caídas bruscas cuando la red encuentra el camino.
6. **Espiral con 16 neuronas.** Con tasa 0.1 se atasca alrededor del 65 %. Baja la tasa a unos 0.03, sube la velocidad y espera: la espiral termina separándose.
7. **Círculo con tasa 1.** La pérdida oscila en vez de bajar: los pasos son tan grandes que la red salta de un lado al otro del valle.

---

## Resultados medidos

Todas las cifras de esta lección se midieron en Node con este mismo `red.js`:

| Prueba | Configuración | Resultado |
|---|---|---|
| XOR (4 puntos) | 2 ocultas, tasa 0.5, hasta 5000 épocas | resuelto 24 de 50 veces |
| XOR (4 puntos) | 3 ocultas | 36 de 50 |
| XOR (4 puntos) | 4 ocultas | 37 de 50 |
| XOR por cuadrantes (160 puntos) | 2 ocultas, 1000 épocas | ≈ 86 % de precisión |
| XOR por cuadrantes | 4 ocultas | 100 % en 10 de 10 |
| Círculo (160 puntos) | 3–4 ocultas | ≈ 93 % |
| Círculo | 8 ocultas | ≈ 99.6 % (100 % en 5 de 8) |
| Espiral (240 puntos) | 16 ocultas, tasa 0.1, 2000 épocas | ≈ 65–70 % |
| Espiral | 16 ocultas, tasa 0.03, 2000 épocas | 100 % en 8 de 8 |
| Comprobación del gradiente | `W1[1][0]`, diferencia central con ε = 1e-4 | numérico 0.05259094 = analítico 0.05259094 |

---

## Ejercicios

1. **Neuronas mínimas.** ¿Cuál es el menor número de neuronas ocultas con el que la red resuelve el XOR por cuadrantes? ¿Y el círculo? Justifícalo mirando las rectas punteadas.
2. **Mide la suerte.** En Node, entrena 50 redes con el XOR de 4 puntos usando 2, 3 y 4 neuronas ocultas. Haz una tabla con cuántas lo resuelven.
3. **Vuelve al escalón.** En `propagar`, cambia `Math.tanh(...)` por un escalón que devuelva 1 o −1, y deja la derivada `(1 - h * h)` de `entrenar` como está. ¿Qué pasa?
4. **Sigmoide en la capa oculta.** Cambia `Math.tanh` por `sigmoide` en `propagar` y la derivada `(1 - h * h)` por `h * (1 - h)` en `entrenar`. Compara cuántas épocas necesita con el círculo.
5. **Tasa extrema.** Usa el círculo con tasa 1 y observa la gráfica de pérdida. ¿Qué significa que oscile?
6. **Verifica el gradiente.** Súmale ε = 0.0001 a un peso de `W1` y calcula la pérdida. Réstale ε y calcúlala otra vez. Calcula `(L₊ − L₋) / (2ε)` y compáralo con `deltaOcultas[j] · entradas[i]`. Si solo haces un ejercicio, haz este: si los números coinciden, entendiste la retropropagación.

<details>
<summary>Soluciones (pistas)</summary>

1. XOR por cuadrantes: 4 en la práctica, dos rectas por cada eje; con 2 neuronas solo llega a ≈ 86 %. Círculo: al menos 3, que forman un triángulo. Cuantas más neuronas, más lados tiene el polígono y más se parece al círculo.
2. Compáralo con la tabla de [Resultados medidos](#resultados-medidos). Más neuronas significa más caminos hacia una solución y menos atascos.
3. Deja de aprender, o aprende al azar. La derivada real del escalón es 0, y `1 − h²` con h = ±1 también da 0, así que ningún peso de `W1` se actualiza.
4. Suele ser más lento. La derivada de la sigmoide vale como máximo 0.25 y la de tanh llega a 1, así que el gradiente llega más débil a la capa oculta.
5. Los pasos son tan grandes que la red salta de un lado al otro del valle sin llegar a bajar.
6. Deben coincidir en 6–8 decimales. Si no coinciden, hay un error en la retropropagación. Esta técnica (*gradient checking*) se usa en la práctica para depurar redes.

</details>
