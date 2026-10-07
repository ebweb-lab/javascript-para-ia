// ============================================================
//  Lección 1 — Una neurona desde cero: el perceptrón
//  Este archivo es el CORAZÓN de la lección. Sin librerías.
// ============================================================

class Perceptron {
  /**
   * @param {number} numEntradas      cuántos números recibe la neurona (aquí: x e y → 2)
   * @param {number} tasaAprendizaje  cuánto corrige en cada error (learning rate)
   */
  constructor(numEntradas, tasaAprendizaje = 0.1) {
    // Pesos iniciales aleatorios entre -1 y 1: la neurona "no sabe nada" al nacer.
    this.pesos = Array.from({ length: numEntradas }, () => Math.random() * 2 - 1);
    this.sesgo = Math.random() * 2 - 1;
    this.tasa = tasaAprendizaje;
  }

  // Paso 1: suma ponderada  z = w1·x1 + w2·x2 + ... + b
  sumaPonderada(entradas) {
    let z = this.sesgo;
    for (let i = 0; i < this.pesos.length; i++) {
      z += this.pesos[i] * entradas[i];
    }
    return z;
  }

  // Paso 2: función de activación "escalón". Convierte z en una decisión.
  activacion(z) {
    return z >= 0 ? 1 : -1;
  }

  // Predicción completa: entradas → suma → activación
  predecir(entradas) {
    return this.activacion(this.sumaPonderada(entradas));
  }

  // Paso 3: aprender de UN ejemplo (regla del perceptrón).
  // error = objetivo - predicción  →  puede valer -2, 0 o 2
  // Si acierta (error 0) no cambia nada. Si falla, empuja los pesos
  // en la dirección que habría dado la respuesta correcta.
  entrenar(entradas, objetivo) {
    const error = objetivo - this.predecir(entradas);
    if (error !== 0) {
      for (let i = 0; i < this.pesos.length; i++) {
        this.pesos[i] += this.tasa * error * entradas[i];
      }
      this.sesgo += this.tasa * error;
    }
    return error;
  }
}
