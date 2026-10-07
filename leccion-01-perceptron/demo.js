// ============================================================
//  Demo interactiva: dibujar puntos y ver aprender al perceptrón
//  Coordenadas del "mundo": x e y van de -1 a 1.
// ============================================================

const canvas = document.getElementById('lienzo');
const ctx = canvas.getContext('2d');

const ui = {
  epoca: document.getElementById('epoca'),
  precision: document.getElementById('precision'),
  w1: document.getElementById('w1'),
  w2: document.getElementById('w2'),
  b: document.getElementById('b'),
  estado: document.getElementById('estado'),
  btnPaso: document.getElementById('btn-paso'),
  btnAuto: document.getElementById('btn-auto'),
  btnReiniciar: document.getElementById('btn-reiniciar'),
  btnPuntos: document.getElementById('btn-puntos'),
  btnLimpiar: document.getElementById('btn-limpiar'),
  tasa: document.getElementById('tasa'),
  tasaValor: document.getElementById('tasa-valor'),
  velocidad: document.getElementById('velocidad'),
  verSecreta: document.getElementById('ver-secreta'),
};

// --- Estado ---------------------------------------------------
let neurona = new Perceptron(2, leerTasa());
let puntos = [];
let secreta = nuevaLineaSecreta();
let epoca = 0;
let indice = 0;            // punto actual dentro de la época
let erroresEpoca = 0;
let auto = false;
let ultimo = null;         // último punto entrenado (para resaltarlo)

// La "verdad" que la neurona debe descubrir: una recta y = m·x + c
function nuevaLineaSecreta() {
  return { m: Math.random() * 2 - 1, c: Math.random() * 0.8 - 0.4 };
}
function etiquetaReal(x, y) {
  return y > secreta.m * x + secreta.c ? 1 : -1;
}

function generarPuntos(n = 80) {
  secreta = nuevaLineaSecreta();
  puntos = Array.from({ length: n }, () => {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    return { x, y, etiqueta: etiquetaReal(x, y) };
  });
  reiniciarEntrenamiento();
}

function reiniciarEntrenamiento() {
  neurona = new Perceptron(2, leerTasa());
  epoca = 0;
  indice = 0;
  erroresEpoca = 0;
  ultimo = null;
  barajar(puntos);
  setEstado('Neurona recién nacida: pesos aleatorios.');
  dibujar();
}

function barajar(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// --- Entrenamiento --------------------------------------------
// Un paso = la neurona mira UN punto y corrige si se equivoca.
function paso() {
  if (puntos.length === 0) return false;
  const p = puntos[indice];
  const error = neurona.entrenar([p.x, p.y], p.etiqueta);
  if (error !== 0) erroresEpoca++;
  ultimo = p;
  indice++;

  // ¿Terminó la época (pasada completa por todos los puntos)?
  if (indice >= puntos.length) {
    epoca++;
    const huboErrores = erroresEpoca;
    indice = 0;
    erroresEpoca = 0;
    barajar(puntos);
    if (huboErrores === 0) {
      setEstado(`✔ Convergió en la época ${epoca}: una pasada entera sin errores.`);
      return true; // terminado
    }
    setEstado(`Época ${epoca}: ${huboErrores} correcciones.`);
    if (epoca >= 500) {
      setEstado('500 épocas sin converger. ¿Tus puntos son separables con una recta?');
      return true;
    }
  }
  return false;
}

function bucle() {
  if (!auto) return;
  const pasos = Number(ui.velocidad.value);
  for (let i = 0; i < pasos; i++) {
    if (paso()) { detenerAuto(); break; }
  }
  dibujar();
  if (auto) requestAnimationFrame(bucle);
}

function detenerAuto() {
  auto = false;
  ui.btnAuto.textContent = '▶ Entrenar';
}

// --- Dibujo ---------------------------------------------------
function css(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

let tam = 0; // lado del canvas en px CSS
function ajustarTamano() {
  const dpr = window.devicePixelRatio || 1;
  tam = canvas.clientWidth;
  canvas.width = tam * dpr;
  canvas.height = tam * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  dibujar();
}

// mundo (-1..1) → píxeles. La y se invierte: en el canvas crece hacia abajo.
const aPx = (x, y) => [((x + 1) / 2) * tam, ((1 - y) / 2) * tam];
const aMundo = (px, py) => [(px / tam) * 2 - 1, 1 - (py / tam) * 2];

function dibujar() {
  if (!tam) return;
  const cA = css('--clase-a'), cB = css('--clase-b');

  // Fondo coloreado según lo que la neurona predice en cada zona
  const celda = tam / 40;
  for (let i = 0; i < 40; i++) {
    for (let j = 0; j < 40; j++) {
      const [x, y] = aMundo(i * celda + celda / 2, j * celda + celda / 2);
      ctx.fillStyle = neurona.predecir([x, y]) === 1 ? css('--zona-a') : css('--zona-b');
      ctx.fillRect(i * celda, j * celda, celda + 1, celda + 1);
    }
  }

  // Ejes
  ctx.strokeStyle = css('--eje');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(tam / 2, 0); ctx.lineTo(tam / 2, tam);
  ctx.moveTo(0, tam / 2); ctx.lineTo(tam, tam / 2);
  ctx.stroke();

  // Línea secreta (la verdad)
  if (ui.verSecreta.checked) {
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = css('--texto-suave');
    ctx.lineWidth = 2;
    linea(-1, secreta.m * -1 + secreta.c, 1, secreta.m + secreta.c);
    ctx.setLineDash([]);
  }

  // Frontera de decisión aprendida: w1·x + w2·y + b = 0  →  y = -(w1·x + b) / w2
  const [w1, w2] = neurona.pesos, b = neurona.sesgo;
  if (Math.abs(w2) > 1e-9) {
    ctx.strokeStyle = css('--texto');
    ctx.lineWidth = 3;
    linea(-1, -(w1 * -1 + b) / w2, 1, -(w1 * 1 + b) / w2);
  }

  // Puntos
  let aciertos = 0;
  for (const p of puntos) {
    const [px, py] = aPx(p.x, p.y);
    const ok = neurona.predecir([p.x, p.y]) === p.etiqueta;
    if (ok) aciertos++;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = p.etiqueta === 1 ? cA : cB;
    ctx.fill();
    // Mal clasificado → anillo de alerta
    if (!ok) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = css('--error');
      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // Último punto entrenado
  if (ultimo) {
    const [px, py] = aPx(ultimo.x, ultimo.y);
    ctx.lineWidth = 2;
    ctx.strokeStyle = css('--texto');
    ctx.beginPath();
    ctx.arc(px, py, 14, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Panel de números
  ui.epoca.textContent = epoca;
  ui.precision.textContent = puntos.length ? Math.round((aciertos / puntos.length) * 100) + '%' : '—';
  ui.w1.textContent = w1.toFixed(3);
  ui.w2.textContent = w2.toFixed(3);
  ui.b.textContent = b.toFixed(3);
}

function linea(x1, y1, x2, y2) {
  const [a, b] = aPx(x1, y1), [c, d] = aPx(x2, y2);
  ctx.beginPath();
  ctx.moveTo(a, b);
  ctx.lineTo(c, d);
  ctx.stroke();
}

function setEstado(t) { ui.estado.textContent = t; }

// El slider es logarítmico: va de -3 a 0 y la tasa es 10^valor (0.001 … 1).
function leerTasa() { return 10 ** Number(ui.tasa.value); }

// --- Eventos --------------------------------------------------
ui.btnPaso.addEventListener('click', () => { detenerAuto(); paso(); dibujar(); });
ui.btnAuto.addEventListener('click', () => {
  auto = !auto;
  ui.btnAuto.textContent = auto ? '⏸ Pausa' : '▶ Entrenar';
  if (auto) bucle();
});
ui.btnReiniciar.addEventListener('click', () => { detenerAuto(); reiniciarEntrenamiento(); });
ui.btnPuntos.addEventListener('click', () => { detenerAuto(); generarPuntos(); });
ui.btnLimpiar.addEventListener('click', () => { detenerAuto(); puntos = []; reiniciarEntrenamiento(); setEstado('Lienzo vacío. Haz clic para añadir puntos.'); });
ui.tasa.addEventListener('input', () => {
  neurona.tasa = leerTasa();
  ui.tasaValor.textContent = neurona.tasa.toPrecision(2);
});
ui.verSecreta.addEventListener('change', dibujar);

// Clic izquierdo → clase A (+1). Clic derecho o Shift+clic → clase B (-1).
function anadirPunto(e, etiqueta) {
  const r = canvas.getBoundingClientRect();
  const [x, y] = aMundo(e.clientX - r.left, e.clientY - r.top);
  puntos.push({ x, y, etiqueta });
  dibujar();
}
canvas.addEventListener('click', (e) => anadirPunto(e, e.shiftKey ? -1 : 1));
canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); anadirPunto(e, -1); });

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', dibujar);
new ResizeObserver(ajustarTamano).observe(canvas);

generarPuntos();
