// --- Utilidades matemáticas -----------------------------------
const aleatorio = () => Math.random() * 2 - 1;

// Sigmoide: aplasta cualquier número al rango (0, 1) → se lee como probabilidad.
const sigmoide = (z) => 1 / (1 + Math.exp(-z));

// Producto escalar: Σ a[i]·b[i]. Es la suma ponderada de la lección 1.
function productoPunto(a, b) {
  let suma = 0;
  for (let i = 0; i < a.length; i++) suma += a[i] * b[i];
  return suma;
}

// Pérdida logarítmica (entropía cruzada binaria): mide lo lejos que está
// la probabilidad p del objetivo t (0 o 1). Si acierta con seguridad → ≈ 0.
function perdida(p, t) {
  const e = 1e-12; // evita log(0)
  return -(t * Math.log(p + e) + (1 - t) * Math.log(1 - p + e));
}

class RedNeuronal {
  /**
   * @param {number} numEntradas  cuántos números entran (aquí: x e y → 2)
   * @param {number} numOcultas   cuántas neuronas tiene la capa oculta
   * @param {number} tasa         tasa de aprendizaje (tamaño del paso)
   */
  constructor(numEntradas, numOcultas, tasa = 0.1) {
    this.tasa = tasa;
    // Capa oculta: cada neurona j tiene sus propios pesos W1[j] y su sesgo b1[j].
    // Es decir: numOcultas perceptrones de la lección 1, uno al lado del otro.
    this.W1 = Array.from({ length: numOcultas }, () =>
      Array.from({ length: numEntradas }, aleatorio));
    this.b1 = Array.from({ length: numOcultas }, aleatorio);
    // Capa de salida: UNA neurona que recibe las salidas de la capa oculta.
    this.W2 = Array.from({ length: numOcultas }, aleatorio);
    this.b2 = aleatorio();
  }

  // Propagación hacia delante: entradas → capa oculta → salida.
  propagar(entradas) {
    // Activación oculta: tanh, curva suave entre -1 y 1 (el escalón ya no sirve:
    // no tiene pendiente y sin pendiente no se puede calcular el gradiente).
    const ocultas = this.W1.map((pesos, j) =>
      Math.tanh(productoPunto(pesos, entradas) + this.b1[j]));
    const salida = sigmoide(productoPunto(this.W2, ocultas) + this.b2);
    return { ocultas, salida };
  }

  predecir(entradas) {
    return this.propagar(entradas).salida;
  }

  // Aprender de UN ejemplo: retropropagación + descenso del gradiente.
  entrenar(entradas, objetivo) {
    const { ocultas, salida } = this.propagar(entradas);

    // 1) Error en la salida. Con sigmoide + pérdida logarítmica, la derivada
    //    de la pérdida respecto a z de salida se simplifica a: salida - objetivo.
    const deltaSalida = salida - objetivo;

    // 2) Retropropagar: cuánta culpa tiene cada neurona oculta.
    //    Regla de la cadena: delta de salida × peso que las une × derivada de tanh.
    //    (Se calcula ANTES de tocar W2, con los pesos que produjeron el error.)
    const deltaOcultas = ocultas.map((h, j) =>
      deltaSalida * this.W2[j] * (1 - h * h));

    // 3) Descenso del gradiente: mover cada peso un paso CONTRA su gradiente.
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
  }
}
