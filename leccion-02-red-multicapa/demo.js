// ============================================================
//  Demo interactiva: una red multicapa aprende formas no lineales
//  Coordenadas del "mundo": x e y van de -1 a 1.
//  Etiquetas: 1 = clase A, 0 = clase B (la salida es una probabilidad).
// ============================================================

const canvas = document.getElementById('lienzo');
const ctx = canvas.getContext('2d');
const grafica = document.getElementById('grafica');
const gctx = grafica.getContext('2d');

const ui = {
  epoca: document.getElementById('epoca'),
  perdida: document.getElementById('perdida'),
  precision: document.getElementById('precision'),
  estado: document.getElementById('estado'),
  btnAuto: document.getElementById('btn-auto'),
  btnEpoca: document.getElementById('btn-epoca'),
  btnReiniciar: document.getElementById('btn-reiniciar'),
  btnLimpiar: document.getElementById('btn-limpiar'),
  datos: document.getElementById('datos'),
  ocultas: document.getElementById('ocultas'),
  ocultasValor: document.getElementById('ocultas-valor'),
  tasa: document.getElementById('tasa'),
  tasaValor: document.getElementById('tasa-valor'),
  velocidad: document.getElementById('velocidad'),
  verOcultas: document.getElementById('ver-ocultas'),
};

// --- Estado ---------------------------------------------------
let red = null;
let puntos = [];
let epoca = 0;
let historial = [];        // pérdida media de cada época (para la gráfica)
let auto = false;

// El slider de tasa es logarítmico: va de -3 a 0 y la tasa es 10^valor (0.001 … 1).
function leerTasa() { return 10 ** Number(ui.tasa.value); }

// --- Datos ----------------------------------------------------
// Cada generador devuelve la etiqueta (1 o 0) de un punto, o crea la lista entera.
const generadores = {
  // XOR: arriba-derecha y abajo-izquierda son B; los otros dos cuadrantes, A.
  xor(n) {
    return Array.from({ length: n }, () => {
      const x = aleatorioLejosDelEje(), y = aleatorioLejosDelEje();
      return { x, y, etiqueta: (x > 0) !== (y > 0) ? 1 : 0 };
    });
  },
  // Círculo: A dentro, B fuera. Ninguna recta sola puede separarlos.
  circulo(n) {
    return Array.from({ length: n }, () => {
      const x = aleatorio(), y = aleatorio();
      return { x, y, etiqueta: x * x + y * y < 0.36 ? 1 : 0 };
    });
  },
  // Espiral: dos brazos enrollados. El reto difícil.
  espiral(n) {
    return Array.from({ length: n }, (_, i) => {
      const etiqueta = i % 2;
      const t = Math.random();
      const angulo = t * 3 * Math.PI + etiqueta * Math.PI;
      const r = 0.1 + t * 0.85;
      return {
        x: r * Math.cos(angulo) + (Math.random() - 0.5) * 0.08,
        y: r * Math.sin(angulo) + (Math.random() - 0.5) * 0.08,
        etiqueta,
      };
    });
  },
};

function aleatorioLejosDelEje() {
  const v = 0.1 + Math.random() * 0.9;
  return Math.random() < 0.5 ? v : -v;
}

function generarPuntos() {
  puntos = generadores[ui.datos.value](ui.datos.value === 'espiral' ? 240 : 160);
  reiniciarRed();
}

function reiniciarRed() {
  red = new RedNeuronal(2, Number(ui.ocultas.value), leerTasa());
  epoca = 0;
  historial = [];
  setEstado(`Red nueva: 2 entradas → ${ui.ocultas.value} ocultas → 1 salida. Pesos aleatorios.`);
  dibujar();
}

function barajar(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// --- Entrenamiento --------------------------------------------
// Una época = la red ve todos los puntos una vez (en orden aleatorio).
function entrenarEpoca() {
  if (puntos.length === 0) return;
  barajar(puntos);
  let suma = 0;
  for (const p of puntos) suma += red.entrenar([p.x, p.y], p.etiqueta);
  historial.push(suma / puntos.length);
  epoca++;
}

function bucle() {
  if (!auto) return;
  const epocas = Number(ui.velocidad.value);
  for (let i = 0; i < epocas; i++) entrenarEpoca();
  dibujar();
  const ultima = historial[historial.length - 1];
  if (ultima !== undefined && ultima < 0.01) {
    detenerAuto();
    setEstado(`✔ Pérdida por debajo de 0.01 en la época ${epoca}. La red ha aprendido la forma.`);
    return;
  }
  requestAnimationFrame(bucle);
}

function detenerAuto() {
  auto = false;
  ui.btnAuto.textContent = '▶ Entrenar';
}

// --- Dibujo ---------------------------------------------------
function css(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

let tam = 0; // lado del canvas principal en px CSS
function ajustarTamano() {
  const dpr = window.devicePixelRatio || 1;
  tam = canvas.clientWidth;
  canvas.width = tam * dpr;
  canvas.height = tam * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  grafica.width = grafica.clientWidth * dpr;
  grafica.height = grafica.clientHeight * dpr;
  gctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  dibujar();
}

// mundo (-1..1) → píxeles. La y se invierte: en el canvas crece hacia abajo.
const aPx = (x, y) => [((x + 1) / 2) * tam, ((1 - y) / 2) * tam];
const aMundo = (px, py) => [(px / tam) * 2 - 1, 1 - (py / tam) * 2];

function dibujar() {
  if (!tam || !red) return;

  // Fondo: probabilidad de clase A en cada celda. Pintamos B y encima A
  // con transparencia = probabilidad → degradado suave donde la red duda.
  const N = 50, celda = tam / N;
  const zonaA = css('--zona-a'), zonaB = css('--zona-b');
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const [x, y] = aMundo(i * celda + celda / 2, j * celda + celda / 2);
      const p = red.predecir([x, y]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = zonaB;
      ctx.fillRect(i * celda, j * celda, celda + 1, celda + 1);
      ctx.globalAlpha = p;
      ctx.fillStyle = zonaA;
      ctx.fillRect(i * celda, j * celda, celda + 1, celda + 1);
    }
  }
  ctx.globalAlpha = 1;

  // Rectas de las neuronas ocultas: cada una es un perceptrón de la lección 1.
  // Su frontera es W1[j][0]·x + W1[j][1]·y + b1[j] = 0.
  if (ui.verOcultas.checked) {
    ctx.strokeStyle = css('--oculta');
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    red.W1.forEach(([w1, w2], j) => {
      const b = red.b1[j];
      if (Math.abs(w2) > 1e-9) linea(-1, -(w1 * -1 + b) / w2, 1, -(w1 + b) / w2);
      else if (Math.abs(w1) > 1e-9) linea(-b / w1, -1, -b / w1, 1);
    });
    ctx.setLineDash([]);
  }

  // Puntos
  let aciertos = 0;
  for (const p of puntos) {
    const [px, py] = aPx(p.x, p.y);
    const prediccion = red.predecir([p.x, p.y]) >= 0.5 ? 1 : 0;
    if (prediccion === p.etiqueta) aciertos++;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = p.etiqueta === 1 ? css('--clase-a') : css('--clase-b');
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = css('--tarjeta');
    ctx.stroke();
    if (prediccion !== p.etiqueta) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = css('--error');
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // Panel de números
  const ultima = historial[historial.length - 1];
  ui.epoca.textContent = epoca;
  ui.perdida.textContent = ultima === undefined ? '—' : ultima.toFixed(4);
  ui.precision.textContent = puntos.length ? Math.round((aciertos / puntos.length) * 100) + '%' : '—';

  dibujarGrafica();
}

// Curva de pérdida: si baja, la red está aprendiendo.
function dibujarGrafica() {
  const w = grafica.clientWidth, h = grafica.clientHeight;
  gctx.clearRect(0, 0, w, h);
  gctx.strokeStyle = css('--eje');
  gctx.lineWidth = 1;
  gctx.beginPath();
  gctx.moveTo(0, h - 0.5); gctx.lineTo(w, h - 0.5);
  gctx.stroke();
  if (historial.length < 2) return;

  const maximo = Math.max(...historial);
  gctx.strokeStyle = css('--acento');
  gctx.lineWidth = 2;
  gctx.beginPath();
  historial.forEach((valor, i) => {
    const x = (i / (historial.length - 1)) * w;
    const y = h - (valor / maximo) * (h - 6) - 1;
    i === 0 ? gctx.moveTo(x, y) : gctx.lineTo(x, y);
  });
  gctx.stroke();
}

function linea(x1, y1, x2, y2) {
  const [a, b] = aPx(x1, y1), [c, d] = aPx(x2, y2);
  ctx.beginPath();
  ctx.moveTo(a, b);
  ctx.lineTo(c, d);
  ctx.stroke();
}

function setEstado(t) { ui.estado.textContent = t; }

// --- Eventos --------------------------------------------------
ui.btnAuto.addEventListener('click', () => {
  auto = !auto;
  ui.btnAuto.textContent = auto ? '⏸ Pausa' : '▶ Entrenar';
  if (auto) bucle();
});
ui.btnEpoca.addEventListener('click', () => { detenerAuto(); entrenarEpoca(); dibujar(); });
ui.btnReiniciar.addEventListener('click', () => { detenerAuto(); reiniciarRed(); });
ui.btnLimpiar.addEventListener('click', () => { detenerAuto(); puntos = []; reiniciarRed(); setEstado('Lienzo vacío. Haz clic para añadir puntos.'); });
ui.datos.addEventListener('change', () => { detenerAuto(); generarPuntos(); });
ui.ocultas.addEventListener('input', () => {
  ui.ocultasValor.textContent = ui.ocultas.value;
  detenerAuto();
  reiniciarRed(); // cambiar la arquitectura obliga a empezar de cero
});
ui.tasa.addEventListener('input', () => {
  red.tasa = leerTasa();
  ui.tasaValor.textContent = red.tasa.toPrecision(2);
});
ui.verOcultas.addEventListener('change', dibujar);

// Clic izquierdo → clase A (1). Clic derecho o Shift+clic → clase B (0).
function anadirPunto(e, etiqueta) {
  const r = canvas.getBoundingClientRect();
  const [x, y] = aMundo(e.clientX - r.left, e.clientY - r.top);
  puntos.push({ x, y, etiqueta });
  dibujar();
}
canvas.addEventListener('click', (e) => anadirPunto(e, e.shiftKey ? 0 : 1));
canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); anadirPunto(e, 0); });

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', dibujar);
new ResizeObserver(ajustarTamano).observe(canvas);

generarPuntos();
