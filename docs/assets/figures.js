/* figures.js — the Epsilon-Hollow project page: one figure per result.

   One small kit, written to the rules of the main site's fig.js:
     - A canvas draws in the same viewBox as the SVG laid over it, and never
       draws a word. Words live in SVG. They may change value in place or
       fade, but they never move.
     - A figure starts its own clock when it arrives on screen, stops when it
       leaves, and starts again from the beginning when it comes back.
     - Under prefers-reduced-motion every figure paints one still frame, its
       end state, and the controls still work.
     - Every verdict is computed, never scripted. The rules are ports of the
       code: certified_beta0's band, certified_top_k's keep, widen and
       fall-back, fold_score's delay cloud and chord quotient, the T4 gain
       margin, voronoi_cap's split at the mean, and T1's minimum separation.
       Where a figure's inputs are chosen for the picture, its caption says so. */
(function () {
  'use strict';

  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var TAU = Math.PI * 2, NS = 'http://www.w3.org/2000/svg';

  /* ================================================================ kit */
  function ease(k) { k = k < 0 ? 0 : k > 1 ? 1 : k; return k * k * k * (k * (k * 6 - 15) + 10); }
  var TOK = null;
  function hex(name, fb) {
    if (!TOK) TOK = getComputedStyle(document.documentElement);
    var h = ((TOK.getPropertyValue(name) || '').trim() || fb).replace('#', '');
    var n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function palette() {
    return {
      ink: hex('--ink', '#1c1b19'), muted: hex('--muted', '#5f5b53'), hair: hex('--hair2', '#cfcbc1'),
      raised: [255, 255, 255], b5: hex('--blue-500', '#2456dc'), v5: hex('--violet-500', '#a66cf0'),
      m5: hex('--mint-500', '#0b93ab'), a5: hex('--amber-500', '#d96a06'), c5: hex('--coral-500', '#d9376e'),
      g5: hex('--green-500', '#146a32')
    };
  }
  function mix(a, b, k) { return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]; }
  function rgb(c, a) { return 'rgba(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ',' + (a < 0 ? 0 : a > 1 ? 1 : a) + ')'; }
  function norm(p) { var l = Math.hypot(p[0], p[1], p[2]); return [p[0] / l, p[1] / l, p[2] / l]; }
  function dist(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]); return Math.sqrt(s); }
  function slerp(a, b, f) {
    var d = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])), w = Math.acos(d);
    if (w < 1e-9) return a.slice();
    var s0 = Math.sin((1 - f) * w) / Math.sin(w), s1 = Math.sin(f * w) / Math.sin(w);
    return [a[0] * s0 + b[0] * s1, a[1] * s0 + b[1] * s1, a[2] * s0 + b[2] * s1];
  }
  function lerpLog(a, b, k) { return Math.pow(10, Math.log10(a) + (Math.log10(b) - Math.log10(a)) * k); }
  /* piecewise keyframes [[t, v], ...], eased, log-interpolated when asked */
  function keyed(K, t, log) {
    if (t <= K[0][0]) return K[0][1];
    for (var i = 1; i < K.length; i++) {
      if (t <= K[i][0]) {
        var k = ease((t - K[i - 1][0]) / Math.max(1, K[i][0] - K[i - 1][0]));
        return log ? lerpLog(K[i - 1][1], K[i][1], k) : K[i - 1][1] + (K[i][1] - K[i - 1][1]) * k;
      }
    }
    return K[K.length - 1][1];
  }

  /* Prim over all pairs: the n - 1 single-linkage merge heights, [i, j, h] */
  function mst(P) {
    var n = P.length, inT = [], best = [], from = [], out = [], i, j;
    for (i = 0; i < n; i++) { inT.push(false); best.push(Infinity); from.push(-1); }
    if (!n) return out;
    best[0] = 0;
    for (var it = 0; it < n; it++) {
      var u = -1;
      for (i = 0; i < n; i++) if (!inT[i] && (u < 0 || best[i] < best[u])) u = i;
      inT[u] = true;
      if (from[u] >= 0) out.push([from[u], u, best[u]]);
      for (j = 0; j < n; j++) {
        if (inT[j]) continue;
        var d = dist(P[u], P[j]);
        if (d < best[j]) { best[j] = d; from[j] = u; }
      }
    }
    return out;
  }
  /* certified_beta0: certified when no merge height lies in [s/sqrt(r), s*sqrt(r)];
     otherwise refused, naming the in-band edge nearest the scale, ties to the earliest */
  function certify(n, E, s, ratio) {
    var r = Math.sqrt(ratio), lo = s / r, hi = s * r, w = null, below = 0;
    for (var i = 0; i < E.length; i++) {
      var h = E[i][2];
      if (h < lo) below++;
      else if (h <= hi && (!w || Math.abs(h - s) < Math.abs(w[2] - s))) w = E[i];
    }
    return { lo: lo, hi: hi, witness: w, count: w ? null : n - below };
  }

  function S(parent, tag, at) {
    var e = document.createElementNS(NS, tag);
    if (at) A(e, at);
    parent.appendChild(e);
    return e;
  }
  function A(e, at) { for (var k in at) { var v = at[k]; if (e.getAttribute(k) !== String(v)) e.setAttribute(k, v); } return e; }
  function setText(el, v) { if (el && el.textContent !== v) el.textContent = v; }
  function phase(root, name) {
    var els = root.querySelectorAll('[data-phase]');
    for (var i = 0; i < els.length; i++) {
      var on = els[i].getAttribute('data-phase') === name ? '1' : '0';
      if (els[i].style.opacity !== on) els[i].style.opacity = on;
    }
  }
  function verdict(svg, text, colour, detail) {
    var v = svg.querySelector('[data-v="verdict"]'), d = svg.querySelector('[data-v="detail"]');
    setText(v, text); if (v) A(v, { fill: colour });
    setText(d, detail || '');
  }
  function fitCanvas(c, vbw, vbh) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var r = c.getBoundingClientRect();
    var w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    if (c._w !== w || c._h !== h || c._d !== dpr) {
      c._w = w; c._h = h; c._d = dpr;
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    var s = Math.min(w / vbw, h / vbh), ox = (w - vbw * s) / 2, oy = (h - vbh * s) / 2;
    var g = c.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, c.width, c.height);
    g.setTransform(s * dpr, 0, 0, s * dpr, ox * dpr, oy * dpr);
    return g;
  }
  /* a control the reader has touched stops its own autoplay */
  function touch(el, state) {
    ['pointerdown', 'keydown', 'input'].forEach(function (ev) {
      el.addEventListener(ev, function () { state.touched = true; });
    });
  }

  var FIGS = [];
  /* register(el, draw, end): draw(t) is called with the figure's own clock
     while el is on screen; under reduced motion it is called once with end */
  function register(el, draw, end) { FIGS.push({ el: el, draw: draw, end: end || 0, born: null }); }

  /* ======================================================== 1 · T1/TSS */
  function figT1(canvas) {
    var svg = canvas.parentNode.querySelector('svg');
    var VBW = 470, VBH = 480, CX = 235, CY = 178, R = 138, TILT = 0.38;
    var AX0 = 40, AX1 = 430, AY = 364, THMIN = 0.50536;
    var tSep = svg.querySelector('[data-v="sep"]');
    var gx = function (r) { return AX0 + r / 1.4 * (AX1 - AX0); };
    var C = palette();
    var tn = Math.acos(1 / Math.sqrt(3)), tl = Math.asin(1 / Math.sqrt(3));
    var sph = function (th, ph) { return [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]; };
    /* the eight, as the code now places them (colatitude), and as the old code
       did (latitude +-t read as colatitude, so the south row lands at phi + pi) */
    var cube = [], bug = [], k;
    for (k = 1; k < 8; k += 2) { cube.push(sph(tn, k * Math.PI / 4)); bug.push(sph(tl, k * Math.PI / 4)); }
    for (k = 1; k < 8; k += 2) { cube.push(sph(Math.PI - tn, k * Math.PI / 4)); bug.push(sph(tl, k * Math.PI / 4 + Math.PI)); }
    var walls = [];
    for (var i = 0; i < 3; i++) {
      var ring = [];
      for (k = 0; k <= 144; k++) {
        var a2 = k / 144 * TAU, c2 = Math.cos(a2), s2 = Math.sin(a2);
        ring.push(i === 0 ? [0, c2, s2] : i === 1 ? [c2, 0, s2] : [c2, s2, 0]);
      }
      walls.push(ring);
    }
    /* the shell: an icosahedron subdivided twice, the main site's hollow
       planet. Its film has no coral: coral means refusal. */
    var gr = (1 + Math.sqrt(5)) / 2;
    var IV = [[-1, gr, 0], [1, gr, 0], [-1, -gr, 0], [1, -gr, 0], [0, -1, gr], [0, 1, gr],
              [0, -1, -gr], [0, 1, -gr], [gr, 0, -1], [gr, 0, 1], [-gr, 0, -1], [-gr, 0, 1]].map(norm);
    var IF = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
              [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
              [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for (var sd = 0; sd < 2; sd++) {
      var mid = {}, NF = [];
      var mp = function (a, b) {
        var key = a < b ? a + '_' + b : b + '_' + a;
        if (mid[key] == null) { mid[key] = IV.length; IV.push(norm([(IV[a][0] + IV[b][0]) / 2, (IV[a][1] + IV[b][1]) / 2, (IV[a][2] + IV[b][2]) / 2])); }
        return mid[key];
      };
      IF.forEach(function (fc) {
        var ab = mp(fc[0], fc[1]), bc = mp(fc[1], fc[2]), ca = mp(fc[2], fc[0]);
        NF.push([fc[0], ab, ca], [fc[1], bc, ab], [fc[2], ca, bc], [ab, bc, ca]);
      });
      IF = NF;
    }
    var seen = {}, SE = [];
    IF.forEach(function (fc) {
      for (var q = 0; q < 3; q++) {
        var a = fc[q], b = fc[(q + 1) % 3], key = a < b ? a * 1000 + b : b * 1000 + a;
        if (!seen[key]) { seen[key] = 1; SE.push(a, b); }
      }
    });
    var stops = [C.m5, C.b5, C.v5, C.b5], iri = [];
    for (var qb = 0; qb < 16; qb++) {
      var pos = qb / 16 * stops.length, si = Math.floor(pos);
      iri.push(mix(stops[si], stops[(si + 1) % stops.length], pos - si));
    }
    var film3 = [C.m5, C.b5, C.v5];
    var spr = (function () {
      var cv = document.createElement('canvas'), Z = 24; cv.width = cv.height = Z;
      var x = cv.getContext('2d'), q = x.createRadialGradient(Z / 2, Z / 2, 0, Z / 2, Z / 2, Z / 2);
      q.addColorStop(0, rgb(mix(C.b5, C.raised, 0.55), 1)); q.addColorStop(0.38, rgb(C.b5, 0.6)); q.addColorStop(1, rgb(C.b5, 0));
      x.fillStyle = q; x.fillRect(0, 0, Z, Z);
      return cv;
    })();

    var CYC = 9000;
    function draw(t) {
      var g = fitCanvas(canvas, VBW, VBH);
      var tc = REDUCED ? 8000 : t % CYC;
      var Al = REDUCED ? 1 : ease(t / 1200);
      var fold = tc < 1600 ? 0 : tc < 3400 ? ease((tc - 1600) / 1800) : tc < 5600 ? 1 : tc < 7400 ? 1 - ease((tc - 5600) / 1800) : 0;
      var cent = cube.map(function (c, n) { return fold > 0 ? norm(slerp(c, bug[n], fold)) : c; });
      var minSep = Infinity, i2, j2;
      for (i2 = 0; i2 < 8; i2++) for (j2 = i2 + 1; j2 < 8; j2++) {
        var d = Math.acos(Math.max(-1, Math.min(1, cent[i2][0] * cent[j2][0] + cent[i2][1] * cent[j2][1] + cent[i2][2] * cent[j2][2])));
        if (d < minSep) minSep = d;
      }
      var ok = minSep >= THMIN;
      phase(svg, ok ? 'ok' : 'bad');
      setText(tSep, minSep.toFixed(4));

      var yaw = REDUCED ? 0.55 : 0.55 + t * TAU / 64000;
      var cy = Math.cos(yaw), sy = Math.sin(yaw), ct = Math.cos(TILT), st = Math.sin(TILT);
      var view = function (p) {
        var u = cy * p[0] - sy * p[1], d0 = -(sy * p[0] + cy * p[1]), w = p[2];
        return [CX + R * u, CY - R * (w * ct - d0 * st), w * st + d0 * ct];
      };
      var air = g.createRadialGradient(CX, CY, R * 0.86, CX, CY, R * 1.14);
      air.addColorStop(0, rgb(C.b5, 0)); air.addColorStop(0.45, rgb(C.b5, 0.08 * Al)); air.addColorStop(1, rgb(C.b5, 0));
      g.fillStyle = air; g.beginPath(); g.arc(CX, CY, R * 1.14, 0, TAU); g.fill();
      var glass = g.createRadialGradient(CX - R * 0.35, CY - R * 0.4, R * 0.1, CX, CY, R);
      glass.addColorStop(0, rgb(C.raised, 0.7 * Al)); glass.addColorStop(1, rgb(C.raised, 0.08 * Al));
      g.fillStyle = glass; g.beginPath(); g.arc(CX, CY, R, 0, TAU); g.fill();
      g.strokeStyle = rgb(C.hair, 0.9 * Al); g.lineWidth = 1;
      g.beginPath(); g.arc(CX, CY, R, 0, TAU); g.stroke();

      var SV = IV.map(view), sb = [], sf = [], q;
      for (q = 0; q < 16; q++) { sb.push([]); sf.push([]); }
      for (q = 0; q < SE.length; q += 2) {
        var p1 = SV[SE[q]], p2 = SV[SE[q + 1]], mz = (p1[2] + p2[2]) / 2;
        var hu = Math.atan2(p1[1] + p2[1] - 2 * CY, p1[0] + p2[0] - 2 * CX) / TAU + 0.25 * mz + (REDUCED ? 0.2 : t / 14000);
        hu -= Math.floor(hu);
        (mz > 0 ? sf : sb)[Math.floor(hu * 16) % 16].push(p1, p2);
      }
      var film = function (B, a) {
        g.lineWidth = 0.8;
        for (var bk = 0; bk < 16; bk++) {
          var L = B[bk]; if (!L.length) continue;
          g.strokeStyle = rgb(iri[bk], a * Al);
          g.beginPath();
          for (var m = 0; m < L.length; m += 2) { g.moveTo(L[m][0], L[m][1]); g.lineTo(L[m + 1][0], L[m + 1][1]); }
          g.stroke();
        }
      };
      var wallA = (1 - fold) * Al;
      var drawWalls = function (front) {
        if (wallA <= 0.01) return;
        for (var wi = 0; wi < 3; wi++) {
          var ring = walls[wi], open = false;
          g.strokeStyle = rgb(film3[wi], (front ? 0.6 : 0.18) * wallA);
          g.lineWidth = front ? 1.4 : 1;
          g.beginPath();
          for (var m = 0; m < ring.length; m++) {
            var v = view(ring[m]);
            if ((v[2] >= 0) === front) { if (!open) { g.moveTo(v[0], v[1]); open = true; } else g.lineTo(v[0], v[1]); }
            else open = false;
          }
          g.stroke();
        }
      };
      var CV = cent.map(view);
      var drawCents = function (front) {
        for (var ci = 0; ci < 8; ci++) {
          var v = CV[ci];
          if ((v[2] >= 0) !== front) continue;
          g.globalAlpha = (front ? 1 : 0.35) * Al;
          g.drawImage(spr, v[0] - 8, v[1] - 8, 16, 16);
          g.globalAlpha = 1;
          g.strokeStyle = rgb(C.ink, (front ? 0.6 : 0.2) * Al); g.lineWidth = 1.2;
          g.beginPath(); g.arc(v[0], v[1], 6, 0, TAU); g.stroke();
        }
      };
      film(sb, 0.07); drawWalls(false); drawCents(false);
      film(sf, 0.16); drawWalls(true); drawCents(true);
      if (!ok) {
        var pulse = REDUCED ? 1 : 0.6 + 0.4 * Math.sin(t / 260);
        for (var cj = 0; cj < 4; cj++) {
          var v = CV[cj];
          g.fillStyle = rgb(C.c5, 0.14 * pulse * Al); g.beginPath(); g.arc(v[0], v[1], 16, 0, TAU); g.fill();
          g.strokeStyle = rgb(C.c5, 0.9 * Al); g.lineWidth = 1.8; g.beginPath(); g.arc(v[0], v[1], 11, 0, TAU); g.stroke();
        }
      }
      /* the gauge: T1 needs the minimum separation at or above theta_min */
      var xt = gx(THMIN);
      g.fillStyle = rgb(C.c5, 0.08 * Al); g.fillRect(AX0, AY - 22, xt - AX0, 30);
      g.strokeStyle = rgb(C.hair, Al); g.lineWidth = 1;
      g.beginPath(); g.moveTo(AX0, AY + 8); g.lineTo(AX1, AY + 8); g.stroke();
      g.strokeStyle = rgb(C.ink, 0.5 * Al); g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(xt, AY - 24); g.lineTo(xt, AY + 8); g.stroke(); g.setLineDash([]);
      [0, 0.5, 1].forEach(function (r) {
        var x = gx(r); g.strokeStyle = rgb(C.muted, 0.6 * Al);
        g.beginPath(); g.moveTo(x, AY + 8); g.lineTo(x, AY + 13); g.stroke();
      });
      var xs = gx(Math.min(1.4, minSep)), cs = ok ? C.g5 : C.c5;
      g.strokeStyle = rgb(cs, Al); g.lineWidth = 2.6;
      g.beginPath(); g.moveTo(xs, AY + 6); g.lineTo(xs, AY - 20); g.stroke();
      g.fillStyle = rgb(cs, Al); g.beginPath(); g.arc(xs, AY - 20, 3.6, 0, TAU); g.fill();
      mark(g, C, 76, 442, ok, C.g5, Al);
      mark(g, C, 76, 466, !ok, C.c5, Al);
    }
    register(canvas, draw, 8000);
  }
  function mark(g, C, x, y, on, col, a) {
    if (on) {
      g.fillStyle = rgb(col, 0.18 * a); g.beginPath(); g.arc(x, y, 9, 0, TAU); g.fill();
      g.fillStyle = rgb(col, a); g.beginPath(); g.arc(x, y, 4.5, 0, TAU); g.fill();
    } else {
      g.strokeStyle = rgb(C.hair, a); g.lineWidth = 1.2; g.beginPath(); g.arc(x, y, 4.5, 0, TAU); g.stroke();
    }
  }

  /* ======================================================== 2 · locate */
  function figLocate(svg) {
    /* two centroids: auto_sized_dimensions(2) is 4 rows by 8 columns.
       Pole view, colatitude 0..1.2 at 118 px per radian */
    var CX = 150, CY = 170, KP = 118, RMAX = 1.2 * KP, gridG = svg.querySelector('.loc-grid'), m;
    S(gridG, 'circle', { cx: CX, cy: CY, r: RMAX, fill: 'none', stroke: 'var(--hair2)' });
    S(gridG, 'circle', { cx: CX, cy: CY, r: Math.PI / 4 * KP, fill: 'none', stroke: 'var(--hair2)', 'stroke-dasharray': '3 4' });
    for (m = 0; m < 8; m++) {
      var ph = m * Math.PI / 4;
      S(gridG, 'line', { x1: CX, y1: CY, x2: CX + RMAX * Math.cos(ph), y2: CY - RMAX * Math.sin(ph), stroke: 'var(--hair2)' });
    }
    function draw(t) {
      var tc = REDUCED ? 6000 : t % 7600;
      phase(svg, tc < 3600 ? 'old' : 'new');
    }
    register(svg, draw, 6000);
  }

  /* ======================================================= 3 · beta_0 */
  function figBeta(root) {
    var svg = root.querySelector('svg'), input = root.querySelector('input'), out = root.querySelector('output');
    var layer = svg.querySelector('.b0-layer');
    var X0 = 30, W = 410, Y0 = 18, AY = 196, L0 = -3, L1 = 0.5;
    var P = [], cl = [[0.08, 0.19, [[0, 0], [0.009, 0.004], [-0.004, 0.009], [0.003, -0.009], [-0.009, -0.003]]],
                      [0.44, 0.09, [[0, 0], [0.009, 0.005], [-0.006, 0.008], [0.005, -0.008]]],
                      [0.93, 0.23, [[0, 0], [0.01, -0.006], [-0.005, 0.01], [0.008, 0.009], [-0.01, -0.004]]],
                      [0.69, 0.25, [[0, 0]]]];
    cl.forEach(function (c) { c[2].forEach(function (o) { P.push([c[0] + o[0], c[1] + o[1]]); }); });
    var E = mst(P);
    var px = function (p) { return [X0 + p[0] * W, Y0 + p[1] * W]; };
    var lx = function (h) { return X0 + (Math.log10(h) - L0) / (L1 - L0) * W; };
    var band = S(layer, 'rect', { y: AY - 20, height: 28, rx: 3 });
    S(layer, 'line', { x1: X0, x2: X0 + W, y1: AY + 8, y2: AY + 8, stroke: 'var(--hair2)' });
    [0.001, 0.01, 0.1, 1].forEach(function (h) { S(layer, 'line', { x1: lx(h), x2: lx(h), y1: AY + 8, y2: AY + 13, stroke: 'var(--muted)', 'stroke-opacity': 0.6 }); });
    var sLine = S(layer, 'line', { y1: AY - 24, y2: AY + 8, stroke: 'var(--ink)', 'stroke-opacity': 0.55, 'stroke-dasharray': '2 3' });
    var edges = E.map(function (e) { var a = px(P[e[0]]), b = px(P[e[1]]); return S(layer, 'line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], 'stroke-linecap': 'round' }); });
    P.forEach(function (p) { var q = px(p); S(layer, 'circle', { cx: q[0], cy: q[1], r: 3.2, fill: 'var(--blue-500)' }); });
    var ticks = E.map(function (e) { var x = lx(e[2]); return S(layer, 'line', { x1: x, x2: x, y1: AY + 6, 'stroke-linecap': 'round' }); });
    var rings = [0, 1].map(function () { return S(layer, 'circle', { r: 7.5, fill: 'none', stroke: 'var(--coral-500)', 'stroke-width': 1.8 }); });

    function update() {
      var s = Math.pow(10, L0 + (L1 - L0) * (+input.value) / 1000);
      var c = certify(P.length, E, s, 10), w = c.witness;
      var xl = Math.max(X0, lx(c.lo)), xh = Math.min(X0 + W, lx(c.hi));
      A(band, { x: xl.toFixed(1), width: Math.max(0, xh - xl).toFixed(1), fill: w ? 'var(--coral-100)' : 'var(--ground2)' });
      A(sLine, { x1: lx(s).toFixed(1), x2: lx(s).toFixed(1) });
      E.forEach(function (e, i) {
        var h = e[2], col = h < c.lo ? 'var(--blue-500)' : h <= c.hi ? 'var(--coral-500)' : 'var(--green-500)', on = e === w;
        A(edges[i], { stroke: col, 'stroke-width': on ? 2.8 : 1.4, 'stroke-opacity': h > c.hi ? 0.25 : 0.9, 'stroke-dasharray': h > c.hi ? '3 4' : 'none' });
        A(ticks[i], { stroke: col, 'stroke-width': on ? 2.6 : 1.4, y2: on ? AY - 18 : AY - 11 });
      });
      rings.forEach(function (r, k) {
        if (!w) { A(r, { visibility: 'hidden' }); return; }
        var q = px(P[w[k]]); A(r, { visibility: 'visible', cx: q[0].toFixed(1), cy: q[1].toFixed(1) });
      });
      var sv = s < 0.01 ? s.toFixed(4) : s < 1 ? s.toFixed(3) : s.toFixed(2);
      if (w) {
        verdict(svg, 'refused: edge (' + w[0] + ', ' + w[1] + ') named', 'var(--coral-700)', 'height ' + w[2].toFixed(4) + ' in the band');
        out.value = 's = ' + sv + ', refused';
      } else {
        verdict(svg, 'β₀ = ' + c.count + ', certified', 'var(--green-700)', 'no height in [' + c.lo.toPrecision(2) + ', ' + c.hi.toPrecision(2) + ']');
        out.value = 's = ' + sv + ', β₀ = ' + c.count;
      }
    }
    var st = { touched: false };
    touch(input, st);
    input.addEventListener('input', update);
    /* the autoplay sweeps s through every regime: all apart, four, refused, one */
    var K = [[0, 0.0016], [1600, 0.0016], [3200, 0.045], [5000, 0.045], [6600, 0.25], [8200, 0.25], [9800, 2.0], [11400, 2.0], [13400, 0.0016]];
    function draw(t) {
      if (st.touched) return;
      var s = REDUCED ? 0.045 : keyed(K, t % 13400, true);
      input.value = Math.round((Math.log10(s) - L0) / (L1 - L0) * 1000);
      update();
    }
    update();
    register(svg, draw, 0);
  }

  /* ======================================================== 4 · top-k */
  function figTopK(root) {
    var svg = root.querySelector('svg'), input = root.querySelector('input'), out = root.querySelector('output');
    var chip = root.querySelector('[data-nan]'), layer = svg.querySelector('.tk-layer');
    var X0 = 60, W = 380, Y0 = 26, DY = 26, KEEP = 2;
    var score = [0.86, 0.70, 0.64, 0.45, 0.31, 0.12], mult = [1.0, 0.8, 1.3, 0.9, 1.1, 0.7];
    var sx = function (v) { return X0 + Math.max(0, Math.min(1, v)) * W; };
    S(layer, 'line', { x1: X0, x2: X0 + W, y1: 184, y2: 184, stroke: 'var(--hair2)' });
    [0, 0.5, 1].forEach(function (v) { S(layer, 'line', { x1: sx(v), x2: sx(v), y1: 184, y2: 189, stroke: 'var(--muted)', 'stroke-opacity': 0.6 }); });
    var cut = S(layer, 'line', { y1: Y0 - 14, y2: Y0 + DY * 5 + 12, stroke: 'var(--ink)', 'stroke-opacity': 0.5, 'stroke-dasharray': '2 3' });
    var bars = [], dots = [], nanT;
    score.forEach(function (v, i) {
      var y = Y0 + i * DY, lab = S(layer, 'text', { x: 20, y: y + 4, 'font-size': 12, fill: 'var(--muted)', 'class': 'mono' });
      lab.textContent = 'k' + i;
      bars.push(S(layer, 'line', { y1: y, y2: y, 'stroke-width': 9, 'stroke-linecap': 'round' }));
      dots.push(S(layer, 'circle', { cx: sx(v), cy: y, r: 3.2, fill: 'var(--ink)' }));
    });
    nanT = S(layer, 'text', { x: X0 + W, y: Y0 + 3 * DY + 4, 'text-anchor': 'end', 'font-size': 12, 'font-weight': 600, fill: 'var(--amber-700)', 'class': 'mono' });
    var nan = false;
    function update() {
      var r = (+input.value) / 1000;
      var lo = score.map(function (v, i) { return v - r * mult[i]; }), hi = score.map(function (v, i) { return v + r * mult[i]; });
      var keep = [], ok, minLo = Math.min(lo[0], lo[1]);
      if (nan) {
        keep = [0, 1, 2, 3, 4, 5]; ok = false;
      } else {
        ok = minLo > Math.max.apply(null, hi.slice(KEEP));
        score.forEach(function (v, i) { if (i < KEEP || (!ok && hi[i] >= minLo)) keep.push(i); });
      }
      score.forEach(function (v, i) {
        var gone = nan && i === 3;
        A(bars[i], { x1: sx(lo[i]).toFixed(1), x2: sx(hi[i]).toFixed(1), visibility: gone ? 'hidden' : 'visible',
                     stroke: nan ? 'var(--amber-500)' : i < KEEP ? 'var(--blue-500)' : keep.indexOf(i) >= 0 ? 'var(--amber-500)' : 'var(--hair2)',
                     'stroke-opacity': keep.indexOf(i) >= 0 ? 0.6 : 0.9 });
        A(dots[i], { visibility: gone ? 'hidden' : 'visible' });
      });
      setText(nanT, nan ? 'NaN' : '');
      A(cut, { x1: sx(minLo).toFixed(1), x2: sx(minLo).toFixed(1), visibility: nan ? 'hidden' : 'visible' });
      var rv = r.toFixed(3);
      if (nan) {
        verdict(svg, 'non-finite score: refused, dense over all 6', 'var(--amber-700)', 'widening cannot bound a score with no enclosure');
        out.value = 'radius ' + rv + ', NaN: dense fallback';
      } else if (ok) {
        verdict(svg, 'certified: keys 0 and 1', 'var(--green-700)', 'both lower ends clear every other upper end');
        out.value = 'radius ' + rv + ', certified';
      } else {
        verdict(svg, 'refused at the boundary: widened to ' + keep.length + ' keys', 'var(--amber-700)', 'keys ' + keep.join(', ') + ' reach the lowest selected lower end');
        out.value = 'radius ' + rv + ', widened to ' + keep.length;
      }
    }
    var st = { touched: false };
    touch(input, st);
    input.addEventListener('input', update);
    chip.addEventListener('click', function () {
      st.touched = true; nan = !nan; chip.setAttribute('aria-pressed', String(nan)); update();
    });
    var K = [[0, 0.012], [1800, 0.012], [3600, 0.045], [5400, 0.045], [7200, 0.1], [9000, 0.1], [10800, 0.012]];
    function draw(t) {
      if (st.touched) return;
      input.value = Math.round((REDUCED ? 0.045 : keyed(K, t % 10800, false)) * 1000);
      update();
    }
    update();
    register(svg, draw, 0);
  }

  /* ====================================================== 5 · stratum */
  /* A port of aether-core trajectory_shape::fold_score. Checked against the
     commits before use: the V 0.3 + 0.01|t - 100| scores 0.390625 at margin
     1.68, and the period-3 staircase scores 0.96875 at margin 1.5 without the
     chord quotient, the reading abc7e0f fixed. */
  function delayCloud(v) { var p = []; for (var t = 2; t < v.length; t++) p.push([v[t], v[t - 1], v[t - 2]]); return p.slice(-64); }
  function arcResample(P) {
    var n = P.length, cum = [0], out = [], seg = 0, k;
    for (k = 1; k < n; k++) cum.push(cum[k - 1] + dist(P[k - 1], P[k]));
    var total = cum[n - 1];
    for (k = 0; k < n; k++) {
      var tg = total * k / (n - 1);
      while (seg + 2 < n && cum[seg + 1] < tg) seg++;
      var a = cum[seg], b = cum[seg + 1], f = b > a ? Math.min(1, Math.max(0, (tg - a) / (b - a))) : 0;
      out.push([0, 1, 2].map(function (j) { return P[seg][j] + f * (P[seg + 1][j] - P[seg][j]); }));
    }
    return out;
  }
  function isMonotone(P) {
    return [1, -1].some(function (sg) {
      return P.every(function (p, i) { return i === 0 || [0, 1, 2].every(function (k) { return sg * (p[k] - P[i - 1][k]) >= 0; }); });
    });
  }
  function foldEdges(R, eps, quotient) {
    var n = R.length, par = [], E = [], i, j, counted = 0;
    for (i = 0; i < n; i++) par.push(i);
    var find = function (x) { while (par[x] !== x) x = par[x]; return x; };
    var near = function (a, b) { return dist(R[a], R[b]) < eps; };
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
      if (quotient && j === i + 2 && near(i, i + 1) && near(i + 1, j)) { E.push([i, j, 'chord']); continue; }
      if (near(i, j)) {
        counted++;
        E.push([i, j, j === i + 1 ? 'path' : 'cross']);
        var a = find(i), b = find(j); if (a !== b) par[a] = b;
      }
    }
    var b0 = 0; for (i = 0; i < n; i++) if (find(i) === i) b0++;
    return { E: E, cyc: Math.max(0, counted - n + b0) };
  }
  function mstMax(P) { var E = mst(P), m = 0; E.forEach(function (e) { if (e[2] > m) m = e[2]; }); return m; }

  function figStratum(root) {
    var svg = root.querySelector('svg'), layer = svg.querySelector('.st-layer'), chips = root.querySelectorAll('[data-mode]');
    var LB = [24, 34, 212, 232], RB = [258, 34, 446, 232];
    var V = [], St = [], x = 10, t;
    for (t = 0; t < 128; t++) V.push(0.3 + 0.01 * Math.abs(t - 100));
    for (t = 0; t < 128; t++) { x -= t % 3 === 0 ? 0.05 : 0.001; St.push(x); }
    function model(series, margin, quotient, cert) {
      var raw = delayCloud(series), R = arcResample(raw), e = mstMax(R);
      if (cert && isMonotone(raw)) return { R: R, E: [], score: 0, cert: true, win: series.slice(-64) };
      var f = foldEdges(R, e * margin, quotient);
      return { R: R, E: f.E, score: Math.min(1, f.cyc / R.length), cert: false, win: series.slice(-64) };
    }
    var M = { v: model(V, 1.68, true, true), old: model(St, 1.5, false, false), now: model(St, 1.68, true, true) };
    [LB, RB].forEach(function (b) { S(layer, 'rect', { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], rx: 8, fill: 'none', stroke: 'var(--hair)' }); });
    var gLoss = S(layer, 'polyline', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.8, 'stroke-linejoin': 'round' });
    var gEdges = S(layer, 'g', {}), gPath = S(layer, 'polyline', { fill: 'none', stroke: 'var(--ink)', 'stroke-opacity': 0.55, 'stroke-width': 1.2 });
    var gDots = S(layer, 'g', {});
    var built = null;
    function build(mode) {
      var m = M[mode];
      var w = m.win, lo = Math.min.apply(null, w), hi = Math.max.apply(null, w);
      var pts = w.map(function (v, i) {
        return (LB[0] + 10 + i / 63 * (LB[2] - LB[0] - 20)).toFixed(1) + ',' + (LB[3] - 10 - (v - lo) / (hi - lo || 1) * (LB[3] - LB[1] - 20)).toFixed(1);
      });
      A(gLoss, { points: pts.join(' ') });
      var pr = m.R.map(function (p) { return [p[0] - p[2], (p[0] + p[1] + p[2]) / 3]; });
      var xs = pr.map(function (p) { return p[0]; }), ys = pr.map(function (p) { return p[1]; });
      var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
      var P = pr.map(function (p) {
        return [RB[0] + 14 + (p[0] - x0) / (x1 - x0 || 1) * (RB[2] - RB[0] - 28), RB[3] - 14 - (p[1] - y0) / (y1 - y0 || 1) * (RB[3] - RB[1] - 28)];
      });
      A(gPath, { points: P.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ') });
      while (gEdges.firstChild) gEdges.removeChild(gEdges.firstChild);
      while (gDots.firstChild) gDots.removeChild(gDots.firstChild);
      m.E.forEach(function (e) {
        if (e[2] === 'path') return;
        var a = P[e[0]], b = P[e[1]];
        S(gEdges, 'line', { x1: a[0].toFixed(1), y1: a[1].toFixed(1), x2: b[0].toFixed(1), y2: b[1].toFixed(1),
          stroke: e[2] === 'chord' ? 'var(--muted)' : 'var(--coral-500)', 'stroke-width': e[2] === 'chord' ? 1 : 1.3,
          'stroke-opacity': e[2] === 'chord' ? 0.5 : 0.75, 'stroke-dasharray': e[2] === 'chord' ? '2 3' : 'none' });
      });
      P.forEach(function (p) { S(gDots, 'circle', { cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: 1.8, fill: mode === 'v' ? 'var(--violet-500)' : 'var(--blue-500)' }); });
      if (mode === 'v') verdict(svg, 'loop_score ' + m.score.toFixed(3) + ': the curve folds back', 'var(--coral-700)', 'coral edges close loops across the two arms; dashed chords are filled triangles');
      else if (mode === 'old') verdict(svg, 'before: loop_score ' + m.score.toFixed(5) + ', called Overfit', 'var(--coral-700)', 'margin 1.5, every corner chord counted as a loop');
      else verdict(svg, 'now: monotone, loop_score certified 0', 'var(--green-700)', 'is_monotone in O(n); no complex is built');
      chips.forEach(function (c) { c.setAttribute('aria-pressed', String(c.getAttribute('data-mode') === mode)); });
      built = mode;
    }
    var st = { touched: false, mode: 'v', t0: 0 };
    chips.forEach(function (c) {
      c.addEventListener('click', function () { st.touched = true; st.mode = c.getAttribute('data-mode'); build(st.mode); reveal(1); });
    });
    function reveal(k) {
      var len = 900;
      A(gLoss, { 'stroke-dasharray': len, 'stroke-dashoffset': ((1 - k) * len).toFixed(0) });
      gEdges.style.opacity = String(Math.max(0, Math.min(1, (k - 0.5) * 2)));
    }
    var ORDER = ['v', 'old', 'now'], SLOT = 5200;
    function draw(t) {
      if (st.touched) return;
      var mode = REDUCED ? 'now' : ORDER[Math.floor(t / SLOT) % 3];
      if (mode !== built) build(mode);
      reveal(REDUCED ? 1 : ease((t % SLOT) / 1600));
    }
    build('v'); reveal(1);
    register(svg, draw, 0);
  }

  /* ==================================================== 6 · foliation */
  function figFoliation(svg) {
    var L = svg.querySelector('.fo-layer');
    var ROOT = [46, 140], B1 = [150, 140], LA = [300, 86], LB = [300, 196];
    var edge = function (a, b) { return S(L, 'line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: 'var(--hair2)', 'stroke-width': 1.5 }); };
    edge(ROOT, B1); edge(B1, LA); var eB = edge(B1, LB);
    var path = function (pts, col) { return S(L, 'polyline', { points: pts.map(function (p) { return p.join(','); }).join(' '), fill: 'none', stroke: col, 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-opacity': 0.55 }); };
    var pA = path([ROOT, B1, LA], 'var(--blue-500)');
    var pBold = path([[ROOT[0], ROOT[1] + 5], [B1[0], B1[1] + 5], [LA[0], LA[1] + 5]], 'var(--coral-500)');
    var pBnew = path([[ROOT[0], ROOT[1] + 5], [B1[0], B1[1] + 5], LB], 'var(--violet-500)');
    S(L, 'circle', { cx: ROOT[0], cy: ROOT[1], r: 8, fill: 'var(--raised)', stroke: 'var(--ink)', 'stroke-width': 1.5 });
    var lab = function (x, y, txt, at) { var e = S(L, 'text', Object.assign({ x: x, y: y, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' }, at || {})); e.textContent = txt; return e; };
    lab(ROOT[0], ROOT[1] + 26, 'root');
    var plaque = function (p, txt) {
      var g = S(L, 'g', {});
      S(g, 'rect', { x: p[0] - 38, y: p[1] - 16, width: 76, height: 32, rx: 9, fill: 'var(--raised)', stroke: 'var(--ink)', 'stroke-width': 1.3 });
      var t = S(g, 'text', { x: p[0], y: p[1] + 4, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--ink)', 'font-weight': 600 }); t.textContent = txt;
      return g;
    };
    plaque(B1, 'block 1'); var gA = plaque(LA, 'block 2'); var gB = plaque(LB, 'block 2′');
    var rB1 = lab(B1[0], B1[1] + 34, 'ref 1', { 'class': 'mono', 'font-size': 11 });
    var rA = lab(LA[0], LA[1] + 34, 'ref 1', { 'class': 'mono', 'font-size': 11 });
    var rB = lab(LB[0], LB[1] + 34, 'ref 1', { 'class': 'mono', 'font-size': 11 });
    lab(LA[0], LA[1] - 26, 'fold_key 0xe0e00501162145bc', { 'class': 'mono', 'font-size': 10, fill: 'var(--ink)' });
    var ringA = S(L, 'rect', { x: LA[0] - 44, y: LA[1] - 22, width: 88, height: 44, rx: 13, fill: 'none', stroke: 'var(--coral-500)', 'stroke-width': 2 });
    var tagA = lab(420, LA[1] + 4, 'seq A', { fill: 'var(--blue-700)', 'font-weight': 600 });
    var tagB = lab(420, LB[1] + 4, 'seq B', { fill: 'var(--violet-700)', 'font-weight': 600 });
    var task = S(L, 'g', {});
    S(task, 'line', { x1: 410, y1: 150, x2: LA[0] + 42, y2: LA[1] + 12, stroke: 'var(--coral-500)', 'stroke-width': 1.6, 'stroke-dasharray': '4 3' });
    task.appendChild(lab(420, 168, 'task 2', { fill: 'var(--coral-700)', 'font-weight': 600 }));
    var X = S(task, 'text', { x: 366, y: 124, 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 700, fill: 'var(--coral-700)' }); X.textContent = '×';
    var CYC = 13000;
    function vis(el, a) { el.style.opacity = String(a); }
    function draw(t) {
      var tc = REDUCED ? 6000 : t % CYC, k;
      var p = tc < 2400 ? 0 : tc < 5000 ? 1 : tc < 7600 ? 2 : tc < 10200 ? 3 : 4;
      k = ease((tc - [0, 2400, 5000, 7600, 10200][p]) / 900);
      vis(pA, p === 0 ? k * 0.55 : 0.55); vis(tagA, 1);
      vis(pBold, p === 1 ? k * 0.7 : 0);
      vis(ringA, p === 1 ? k : 0);
      vis(pBnew, p === 2 ? k * 0.55 : 0);
      var bAlive = p === 2 ? k : p === 3 ? 1 - k : 0;
      vis(gB, bAlive); vis(rB, bAlive); vis(eB, bAlive); vis(tagB, p === 1 || p === 2 ? 1 : p === 3 ? 1 - k : 0);
      gB.setAttribute('transform', 'translate(' + LB[0] + ' ' + LB[1] + ') scale(' + (0.4 + 0.6 * bAlive).toFixed(3) + ') translate(' + (-LB[0]) + ' ' + (-LB[1]) + ')');
      vis(task, p === 4 ? k : 0);
      setText(rB1, p === 1 || p === 2 ? 'ref 2' : 'ref 1');
      setText(rA, p === 1 ? 'ref 2' : 'ref 1');
      setText(rB, p === 3 ? 'ref 0' : 'ref 1');
      if (p === 0) verdict(svg, 'seq A descends and caches its blocks', 'var(--ink)', 'a plaque\'s refcount is the number of sequences through it');
      else if (p === 1) verdict(svg, 'before: seq B\'s key collides, and it shares A\'s leaf', 'var(--coral-700)', 'different tokens, same fold_key: each reads the other\'s KV state');
      else if (p === 2) verdict(svg, 'now: tokens are compared, and seq B gets its own leaf', 'var(--green-700)', 'a descent shares a plaque only on an exact token match');
      else if (p === 3) verdict(svg, 'seq B releases: its leaf is a free face and collapses', 'var(--ink)', 'resident, refcount 0, no resident children: the only admissible victim');
      else verdict(svg, 'another task releases seq A: refused, NoSuchSeq', 'var(--coral-700)', 'ENOENT, as for an unused id; A\'s blocks stay referenced');
    }
    register(svg, draw, 6000);
  }

  /* ===================================================== 7 · governor */
  function figGovernor(root) {
    var svg = root.querySelector('svg'), input = root.querySelector('input'), out = root.querySelector('output');
    var L = svg.querySelector('.gv-layer');
    var ALPHA = 0.01, BETA = 0.05, X0 = 50, X1 = 430, D0 = -2, D1 = 0.5, Y0 = 24, Y1 = 184, M0 = -1.7, M1 = 1;
    var gx = function (dt) { return X0 + (Math.log10(dt) - D0) / (D1 - D0) * (X1 - X0); };
    var gy = function (m) { return Y1 - (Math.log10(m) - M0) / (M1 - M0) * (Y1 - Y0); };
    var margin = function (dt) { return ALPHA + BETA / dt; };
    S(L, 'rect', { x: X0, y: gy(1), width: X1 - X0, height: Y1 - gy(1), fill: 'var(--green-100)', 'fill-opacity': 0.6 });
    S(L, 'rect', { x: X0, y: Y0, width: X1 - X0, height: gy(1) - Y0, fill: 'var(--coral-100)', 'fill-opacity': 0.5 });
    S(L, 'line', { x1: X0, x2: X1, y1: gy(1), y2: gy(1), stroke: 'var(--ink)', 'stroke-opacity': 0.5, 'stroke-dasharray': '3 3' });
    var one = S(L, 'text', { x: X0 - 6, y: gy(1) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)', 'class': 'mono' }); one.textContent = '1';
    S(L, 'line', { x1: X0, x2: X1, y1: Y1, y2: Y1, stroke: 'var(--hair2)' });
    [0.01, 0.1, 1].forEach(function (d) { S(L, 'line', { x1: gx(d), x2: gx(d), y1: Y1, y2: Y1 + 5, stroke: 'var(--muted)', 'stroke-opacity': 0.6 }); });
    var pts = [];
    for (var i = 0; i <= 80; i++) {
      var dt = Math.pow(10, D0 + (D1 - D0) * i / 80), m = margin(dt);
      pts.push(gx(dt).toFixed(1) + ',' + Math.max(Y0, Math.min(Y1, gy(m))).toFixed(1));
    }
    S(L, 'polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 });
    var fixed = function (dt, col, txt, dx, anchor) {
      S(L, 'circle', { cx: gx(dt), cy: gy(margin(dt)), r: 5, fill: col });
      var e = S(L, 'text', { x: gx(dt) + dx, y: gy(margin(dt)) - 9, 'text-anchor': anchor, 'font-size': 11, fill: col, 'class': 'mono', 'font-weight': 600 }); e.textContent = txt;
    };
    fixed(0.01, 'var(--coral-700)', 'runtime dt 0.01: 5.01', 8, 'start');
    fixed(1, 'var(--green-700)', 'dt 1: 0.06', 0, 'middle');
    var cur = S(L, 'circle', { r: 8, fill: 'none', stroke: 'var(--blue-500)', 'stroke-width': 2.2 });
    /* |e| of the run 6c450d4 quotes: 0.1, 0.0959612, 0.0960443 */
    var E3 = [0.1, 0.0959612, 0.0960443], ex = function (k) { return 70 + k * 80; }, ey = function (v) { return 316 - (v - 0.0955) / 0.0048 * 36; };
    S(L, 'polyline', { points: E3.map(function (v, k) { return ex(k) + ',' + ey(v).toFixed(1); }).join(' '), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.5 });
    E3.forEach(function (v, k) {
      S(L, 'circle', { cx: ex(k), cy: ey(v), r: 4, fill: k === 2 ? 'var(--coral-500)' : 'var(--ink)' });
      var e = S(L, 'text', { x: ex(k), y: ey(v) - 9, 'text-anchor': 'middle', 'font-size': 10, fill: k === 2 ? 'var(--coral-700)' : 'var(--muted)', 'class': 'mono' });
      e.textContent = v.toFixed(k ? 7 : 1);
    });
    function update() {
      var dt = Math.pow(10, D0 + (D1 - D0) * (+input.value) / 1000), m = margin(dt), ok = m < 1;
      A(cur, { cx: gx(dt).toFixed(1), cy: Math.max(Y0, Math.min(Y1, gy(m))).toFixed(1) });
      var dv = dt < 0.1 ? dt.toFixed(3) : dt.toFixed(2);
      verdict(svg, 'dt ' + dv + ': α + β/dt = ' + m.toFixed(3) + (ok ? ', certified' : ', not certified'), ok ? 'var(--green-700)' : 'var(--coral-700)');
      out.value = 'dt = ' + dv + ', margin ' + m.toFixed(3) + (ok ? ', certified' : ', not certified');
    }
    var st = { touched: false };
    touch(input, st);
    input.addEventListener('input', update);
    var K = [[0, 0.01], [1800, 0.01], [5200, 1], [7000, 1], [10400, 0.01]];
    function draw(t) {
      if (st.touched) return;
      var dt = REDUCED ? 0.01 : keyed(K, t % 10400, true);
      input.value = Math.round((Math.log10(dt) - D0) / (D1 - D0) * 1000);
      update();
    }
    update();
    register(svg, draw, 0);
  }

  /* ======================================================== 8 · cells */
  function figCells(svg) {
    var L = svg.querySelector('.ce-layer'), X0 = 40, W = 390, AY = 176, N = 65;
    var seed = 20260925, rnd = function () { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
    var u = [], h = [], i;
    for (i = 0; i < N; i++) { u.push(Math.max(-1, Math.min(1, (rnd() + rnd() + rnd()) / 1.5 - 1))); h.push(rnd()); }
    var mean = u.reduce(function (a, b) { return a + b; }, 0) / N;      /* split_cell: boundary at the mean */
    var ux = function (v) { return X0 + (v + 1) / 2 * W; };
    S(L, 'line', { x1: X0, x2: X0 + W, y1: AY, y2: AY, stroke: 'var(--hair2)' });
    [-1, 0, 1].forEach(function (v) {
      S(L, 'line', { x1: ux(v), x2: ux(v), y1: AY, y2: AY + 5, stroke: 'var(--muted)', 'stroke-opacity': 0.6 });
      var e = S(L, 'text', { x: ux(v), y: AY + 17, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)', 'class': 'mono' }); e.textContent = String(v);
    });
    var bnd = S(L, 'line', { x1: ux(mean), x2: ux(mean), y1: 30, y2: AY, stroke: 'var(--ink)', 'stroke-opacity': 0.55, 'stroke-dasharray': '3 3' });
    var bl = S(L, 'text', { x: ux(mean), y: 22, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink)' }); bl.textContent = 'split at the mean';
    /* stack the dots so none overlaps */
    var pos = [], dots = [], rings = [];
    for (i = 0; i < N; i++) {
      var x = ux(u[i]), lvl = 0;
      for (var j = 0; j < i; j++) if (Math.abs(pos[j][0] - x) < 7.5 && pos[j][2] === lvl) { lvl++; j = -1; }
      pos.push([x, AY - 9 - lvl * 8.5, lvl]);
    }
    for (i = 0; i < N; i++) {
      rings.push(S(L, 'circle', { cx: pos[i][0].toFixed(1), cy: pos[i][1].toFixed(1), r: 6.2, fill: 'none', stroke: 'var(--coral-500)', 'stroke-width': 1.6 }));
      dots.push(S(L, 'circle', { cx: pos[i][0].toFixed(1), cy: pos[i][1].toFixed(1), r: 3.4 }));
    }
    var sideOf = function (v) { return v > mean ? 1 : 0; };                  /* side(): 1 above the boundary */
    var lost = 0;
    for (i = 0; i < N; i++) if (sideOf(h[i]) !== sideOf(u[i])) lost++;
    function draw(t) {
      var tc = REDUCED ? 9000 : t % 11200;
      var shown = tc < 3200 ? Math.floor(tc / 3200 * N) + 1 : N;
      var p = tc < 3500 ? 0 : tc < 7200 ? 1 : 2;
      for (var q = 0; q < N; q++) {
        var col = p === 0 ? 'var(--blue-500)' : (p === 1 ? sideOf(h[q]) : sideOf(u[q])) ? 'var(--violet-500)' : 'var(--mint-500)';
        A(dots[q], { fill: col, visibility: q < shown ? 'visible' : 'hidden' });
        A(rings[q], { visibility: p === 1 && sideOf(h[q]) !== sideOf(u[q]) ? 'visible' : 'hidden' });
      }
      bnd.style.opacity = bl.style.opacity = p === 0 ? '0' : '1';
      if (p === 0) verdict(svg, shown > 64 ? 'the 65th insert: the cell splits' : shown + ' files in one cell', 'var(--ink)', 'a cell holds 64 before it splits');
      else if (p === 1) verdict(svg, 'before: split by a hash of the inode id', 'var(--coral-700)', lost + ' of 65 files sit on the other side from where locate looks');
      else verdict(svg, 'now: split and located by the same point', 'var(--green-700)', 'every id is exactly once in its locate() bucket');
    }
    register(svg, draw, 9000);
  }

  /* ===================================================== 9 · boot log */
  function figBoot(pre) {
    var lines = Array.prototype.slice.call(pre.querySelectorAll('span'));
    if (REDUCED) return;
    lines.forEach(function (l) { l.style.visibility = 'hidden'; });
    function draw(t) {
      var n = Math.floor((t - 200) / 190);
      for (var i = 0; i < lines.length; i++) {
        lines[i].style.visibility = i <= n ? 'visible' : 'hidden';
        lines[i].classList.toggle('cur', i === Math.min(n, lines.length - 1));
      }
    }
    register(pre, draw, 99999);
  }

  /* ================================================= shared sphere */
  /* the eight boot centroids (cube vertices, colatitude acos(1/sqrt3) and
     pi - acos(1/sqrt3)), the icosphere film, and a painter for a glass S2 */
  function cubeCentroids() {
    var tn = Math.acos(1 / Math.sqrt(3)), out = [], k;
    var sph = function (th, ph) { return [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]; };
    for (k = 1; k < 8; k += 2) out.push(sph(tn, k * Math.PI / 4));
    for (k = 1; k < 8; k += 2) out.push(sph(Math.PI - tn, k * Math.PI / 4));
    return out;
  }
  function icosphere(C) {
    var gr = (1 + Math.sqrt(5)) / 2;
    var IV = [[-1, gr, 0], [1, gr, 0], [-1, -gr, 0], [1, -gr, 0], [0, -1, gr], [0, 1, gr],
              [0, -1, -gr], [0, 1, -gr], [gr, 0, -1], [gr, 0, 1], [-gr, 0, -1], [-gr, 0, 1]].map(norm);
    var IF = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
              [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
              [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for (var sd = 0; sd < 2; sd++) {
      var mid = {}, NF = [];
      var mp = function (a, b) {
        var key = a < b ? a + '_' + b : b + '_' + a;
        if (mid[key] == null) { mid[key] = IV.length; IV.push(norm([(IV[a][0] + IV[b][0]) / 2, (IV[a][1] + IV[b][1]) / 2, (IV[a][2] + IV[b][2]) / 2])); }
        return mid[key];
      };
      IF.forEach(function (fc) {
        var ab = mp(fc[0], fc[1]), bc = mp(fc[1], fc[2]), ca = mp(fc[2], fc[0]);
        NF.push([fc[0], ab, ca], [fc[1], bc, ab], [fc[2], ca, bc], [ab, bc, ca]);
      });
      IF = NF;
    }
    var seen = {}, SE = [];
    IF.forEach(function (fc) {
      for (var q = 0; q < 3; q++) {
        var a = fc[q], b = fc[(q + 1) % 3], key = a < b ? a * 1000 + b : b * 1000 + a;
        if (!seen[key]) { seen[key] = 1; SE.push(a, b); }
      }
    });
    var stops = [C.m5, C.b5, C.v5, C.b5], iri = [];
    for (var qb = 0; qb < 16; qb++) {
      var pos = qb / 16 * stops.length, si = Math.floor(pos);
      iri.push(mix(stops[si], stops[(si + 1) % stops.length], pos - si));
    }
    var walls = [];
    for (var i = 0; i < 3; i++) {
      var ring = [];
      for (var k = 0; k <= 96; k++) {
        var a2 = k / 96 * TAU, c2 = Math.cos(a2), s2 = Math.sin(a2);
        ring.push(i === 0 ? [0, c2, s2] : i === 1 ? [c2, 0, s2] : [c2, s2, 0]);
      }
      walls.push(ring);
    }
    return { IV: IV, SE: SE, iri: iri, walls: walls };
  }
  function glowSprite(base, C) {
    var cv = document.createElement('canvas'), Z = 24; cv.width = cv.height = Z;
    var x = cv.getContext('2d'), q = x.createRadialGradient(Z / 2, Z / 2, 0, Z / 2, Z / 2, Z / 2);
    q.addColorStop(0, rgb(mix(base, C.raised, 0.55), 1)); q.addColorStop(0.38, rgb(base, 0.6)); q.addColorStop(1, rgb(base, 0));
    x.fillStyle = q; x.fillRect(0, 0, Z, Z);
    return cv;
  }
  /* paint a glass sphere at (cx, cy, R) turned by yaw and tilted; returns
     view() so callers can place their own points on it */
  function paintSphere(g, C, I, cx, cy, R, yaw, tilt, drift, a, wallCols) {
    var cy_ = Math.cos(yaw), sy = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt);
    var view = function (p) {
      var u = cy_ * p[0] - sy * p[1], d0 = -(sy * p[0] + cy_ * p[1]), w = p[2];
      return [cx + R * u, cy - R * (w * ct - d0 * st), w * st + d0 * ct];
    };
    var air = g.createRadialGradient(cx, cy, R * 0.86, cx, cy, R * 1.16);
    air.addColorStop(0, rgb(C.b5, 0)); air.addColorStop(0.45, rgb(C.b5, 0.1 * a)); air.addColorStop(1, rgb(C.b5, 0));
    g.fillStyle = air; g.beginPath(); g.arc(cx, cy, R * 1.16, 0, TAU); g.fill();
    var glass = g.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R);
    glass.addColorStop(0, rgb(C.raised, 0.85 * a)); glass.addColorStop(1, rgb(C.raised, 0.18 * a));
    g.fillStyle = glass; g.beginPath(); g.arc(cx, cy, R, 0, TAU); g.fill();
    g.strokeStyle = rgb(C.hair, a); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, R, 0, TAU); g.stroke();
    var SV = I.IV.map(view), sb = [], sf = [], q;
    for (q = 0; q < 16; q++) { sb.push([]); sf.push([]); }
    for (q = 0; q < I.SE.length; q += 2) {
      var p1 = SV[I.SE[q]], p2 = SV[I.SE[q + 1]], mz = (p1[2] + p2[2]) / 2;
      var hu = Math.atan2(p1[1] + p2[1] - 2 * cy, p1[0] + p2[0] - 2 * cx) / TAU + 0.25 * mz + drift;
      hu -= Math.floor(hu);
      (mz > 0 ? sf : sb)[Math.floor(hu * 16) % 16].push(p1, p2);
    }
    var film = function (B, al) {
      g.lineWidth = 0.8;
      for (var bk = 0; bk < 16; bk++) {
        var L = B[bk]; if (!L.length) continue;
        g.strokeStyle = rgb(I.iri[bk], al * a);
        g.beginPath();
        for (var m = 0; m < L.length; m += 2) { g.moveTo(L[m][0], L[m][1]); g.lineTo(L[m + 1][0], L[m + 1][1]); }
        g.stroke();
      }
    };
    var walls = function (front) {
      for (var wi = 0; wi < 3; wi++) {
        var ring = I.walls[wi], open = false;
        g.strokeStyle = rgb(wallCols[wi], (front ? 0.6 : 0.16) * a); g.lineWidth = front ? 1.4 : 1;
        g.beginPath();
        for (var m = 0; m < ring.length; m++) {
          var v = view(ring[m]);
          if ((v[2] >= 0) === front) { if (!open) { g.moveTo(v[0], v[1]); open = true; } else g.lineTo(v[0], v[1]); }
          else open = false;
        }
        g.stroke();
      }
    };
    return { view: view, back: function () { film(sb, 0.08); walls(false); }, front: function () { film(sf, 0.18); walls(true); } };
  }

  /* ======================================================== the seal */
  function figSeal(canvas) {
    var C = palette(), I = icosphere(C), CENT = cubeCentroids();
    var cols = [C.b5, C.v5, C.m5, C.a5];
    var spr = cols.map(function (c) { return glowSprite(c, C); });
    var VBW = 480, VBH = 520, NX = 292, NY = 222, BR = 86;
    var bodyTop = mix(C.b5, C.raised, 0.78), bodyBot = mix(C.m5, C.raised, 0.8), ink = C.ink;
    var P = new Path2D('M292 222 C304 232 306 256 300 276 C296 290 300 302 312 318 C340 350 356 404 344 446 C334 478 300 494 250 494 C200 494 160 488 134 476 C150 440 176 392 200 356 C214 334 216 304 222 280 C228 250 250 228 272 220 C280 216 286 216 292 222 Z');
    var BELLY = new Path2D('M306 304 C336 340 348 400 332 450 C318 472 292 482 262 482 C300 442 312 384 300 322 Z');
    var FLIP = new Path2D('M314 370 C344 390 366 420 374 448 C352 442 330 426 310 404 Z');
    var TAIL = new Path2D('M140 472 C112 460 86 454 60 460 C78 473 100 481 122 483 C100 491 84 503 78 514 C104 512 128 499 148 484 Z');
    function draw(t) {
      var g = fitCanvas(canvas, VBW, VBH);
      var tt = REDUCED ? 0 : t, Al = REDUCED ? 1 : ease(t / 1000);
      var bob = REDUCED ? 0 : 3 * Math.sin(tt * TAU / 2600);
      var sway = REDUCED ? 0 : 0.045 * Math.sin(tt * TAU / 3400);
      var bc = tt % 4600, open = bc > 4440 ? Math.abs(bc - 4520) / 80 : 1;
      if (!REDUCED && (tt % 13800) > 13500) open = Math.min(open, Math.abs((tt % 13800) - 13650) / 150);
      /* shadow, which does not bob */
      var sh = g.createRadialGradient(236, 500, 4, 236, 500, 130);
      sh.addColorStop(0, rgb(ink, 0.12 * Al)); sh.addColorStop(1, rgb(ink, 0));
      g.fillStyle = sh; g.beginPath(); g.ellipse(236, 500, 130, 12, 0, 0, TAU); g.fill();
      g.save(); g.translate(0, bob);
      g.lineJoin = 'round'; g.lineCap = 'round';
      /* tail and flipper behind the body */
      var gr = g.createLinearGradient(180, 220, 320, 500);
      gr.addColorStop(0, rgb(bodyTop, Al)); gr.addColorStop(1, rgb(bodyBot, Al));
      g.fillStyle = gr; g.strokeStyle = rgb(ink, 0.85 * Al); g.lineWidth = 2.2;
      g.fill(TAIL); g.stroke(TAIL);
      g.fill(P); g.stroke(P);
      g.fillStyle = rgb(C.raised, 0.55 * Al); g.fill(BELLY);
      /* a few spots on the back */
      [[214, 380, 4], [196, 420, 3], [230, 342, 3], [178, 452, 3.5], [236, 410, 2.5]].forEach(function (s) {
        g.fillStyle = rgb(mix(C.b5, C.raised, 0.45), 0.45 * Al); g.beginPath(); g.arc(s[0], s[1], s[2], 0, TAU); g.fill();
      });
      g.fillStyle = gr; g.fill(FLIP); g.stroke(FLIP);
      /* face */
      g.fillStyle = rgb(C.c5, 0.16 * Al); g.beginPath(); g.ellipse(282, 262, 11, 7, 0.3, 0, TAU); g.fill();
      g.fillStyle = rgb(ink, Al);
      g.beginPath(); g.ellipse(262, 246, 6.5, Math.max(0.8, 7.5 * open), 0, 0, TAU); g.fill();
      if (open > 0.5) { g.fillStyle = rgb(C.raised, Al); g.beginPath(); g.arc(264.5, 243, 2, 0, TAU); g.fill(); }
      g.fillStyle = rgb(ink, Al); g.beginPath(); g.ellipse(291, 225, 7, 5, 0.5, 0, TAU); g.fill();
      g.strokeStyle = rgb(ink, 0.8 * Al); g.lineWidth = 1.8;
      g.beginPath(); g.moveTo(297, 252); g.quadraticCurveTo(292, 260, 284, 258); g.stroke();
      g.strokeStyle = rgb(ink, 0.45 * Al); g.lineWidth = 1.2;
      [[300, 240, 334, 228], [302, 246, 338, 244], [300, 252, 332, 260]].forEach(function (w) {
        g.beginPath(); g.moveTo(w[0], w[1]); g.quadraticCurveTo((w[0] + w[2]) / 2, (w[1] + w[3]) / 2 - 3, w[2], w[3]); g.stroke();
      });
      /* the sphere, balanced on the nose: it sways about the nose tip */
      g.translate(NX, NY); g.rotate(sway); g.translate(-NX, -NY);
      var S2 = paintSphere(g, C, I, NX, NY - BR - 4, BR, 0.55 + tt * TAU / 30000, 0.38, tt / 14000, Al, [C.m5, C.b5, C.v5]);
      S2.back();
      var V = CENT.map(S2.view);
      var dots = function (front) {
        for (var i = 0; i < 8; i++) {
          if ((V[i][2] >= 0) !== front) continue;
          g.globalAlpha = (front ? 1 : 0.35) * Al;
          g.drawImage(spr[i % 4], V[i][0] - 9, V[i][1] - 9, 18, 18);
          g.globalAlpha = 1;
        }
      };
      dots(false); S2.front(); dots(true);
      g.restore();
    }
    register(canvas, draw, 0);
  }

  /* ================================================= pillar figures */
  function pillarStratum(canvas) {
    var svg = canvas.parentNode.querySelector('svg'), C = palette();
    var V = [], D = [], t;
    for (t = 0; t < 128; t++) V.push(0.3 + 0.01 * Math.abs(t - 100));
    for (t = 0; t < 128; t++) D.push(0.05 + 0.55 * Math.exp(-t / 22));
    function model(series) {
      var raw = delayCloud(series), R = arcResample(raw), e = mstMax(R), win = series.slice(-64);
      if (isMonotone(raw)) return { R: R, E: [], score: 0, win: win, mono: true };
      var f = foldEdges(R, e * 1.68, true);
      return { R: R, E: f.E, score: Math.min(1, f.cyc / R.length), win: win, mono: false };
    }
    var M = [model(V), model(D)];
    function layout(m) {
      var lo = Math.min.apply(null, m.win), hi = Math.max.apply(null, m.win);
      m.curve = m.win.map(function (v, i) { return [16 + i / 63 * 128, 160 - (v - lo) / (hi - lo) * 118]; });
      var pr = m.R.map(function (p) { return [p[0] - p[2], (p[0] + p[1] + p[2]) / 3]; });
      var xs = pr.map(function (p) { return p[0]; }), ys = pr.map(function (p) { return p[1]; });
      var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
      m.cloud = pr.map(function (p) { return [180 + (p[0] - x0) / (x1 - x0 || 1) * 124, 160 - (p[1] - y0) / (y1 - y0 || 1) * 118]; });
    }
    M.forEach(layout);
    var SLOT = 5600;
    function draw(t) {
      var g = fitCanvas(canvas, 320, 190), which = REDUCED ? 0 : Math.floor(t / SLOT) % 2, m = M[which];
      var k = REDUCED ? 1 : ease((t % SLOT) / 2600), n = Math.max(2, Math.round(k * m.curve.length));
      var col = which ? C.b5 : C.v5;
      g.lineJoin = 'round'; g.lineCap = 'round';
      g.strokeStyle = rgb(col, 0.95); g.lineWidth = 2.2; g.beginPath();
      for (var i = 0; i < n; i++) { var p = m.curve[i]; if (i) g.lineTo(p[0], p[1]); else g.moveTo(p[0], p[1]); }
      g.stroke();
      var hd = m.curve[n - 1]; g.fillStyle = rgb(col, 1); g.beginPath(); g.arc(hd[0], hd[1], 3.5, 0, TAU); g.fill();
      var cn = Math.max(2, Math.round(k * m.cloud.length));
      g.strokeStyle = rgb(C.ink, 0.45); g.lineWidth = 1.1; g.beginPath();
      for (i = 0; i < cn; i++) { p = m.cloud[i]; if (i) g.lineTo(p[0], p[1]); else g.moveTo(p[0], p[1]); }
      g.stroke();
      var ea = REDUCED ? 1 : Math.max(0, Math.min(1, ((t % SLOT) - 2400) / 900));
      if (ea > 0 && !m.mono) {
        /* the loop the fold closes, shaded, with the edges that close it */
        g.fillStyle = rgb(C.c5, 0.12 * ea); g.beginPath();
        m.cloud.forEach(function (q, j) { if (j) g.lineTo(q[0], q[1]); else g.moveTo(q[0], q[1]); });
        g.closePath(); g.fill();
        m.E.forEach(function (e) {
          if (e[2] !== 'cross') return;
          var a = m.cloud[e[0]], b = m.cloud[e[1]];
          g.strokeStyle = rgb(C.c5, 0.3 * ea); g.lineWidth = 1;
          g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
        });
      }
      for (i = 0; i < cn; i++) { p = m.cloud[i]; g.fillStyle = rgb(col, 0.9); g.beginPath(); g.arc(p[0], p[1], 1.7, 0, TAU); g.fill(); }
      verdict(svg, which ? 'only falls: loop_score certified 0' : 'folds back: loop_score ' + m.score.toFixed(3), which ? 'var(--green-700)' : 'var(--coral-700)');
    }
    register(canvas, draw, 0);
  }

  function pillarFoliation(canvas) {
    var svg = canvas.parentNode.querySelector('svg'), C = palette();
    var N = { r: [34, 104], a: [104, 70], b: [104, 142], a1: [184, 44], a2: [184, 96], b1: [184, 142], a11: [264, 36], a12: [264, 76], a21: [264, 118] };
    var E = [['r', 'a'], ['r', 'b'], ['a', 'a1'], ['a', 'a2'], ['b', 'b1'], ['a1', 'a11'], ['a1', 'a12'], ['a2', 'a21']];
    var SEQ = [[['r', 'a', 'a1', 'a11'], C.b5], [['r', 'a', 'a1', 'a12'], C.v5], [['r', 'a', 'a2', 'a21'], C.m5], [['r', 'b', 'b1'], C.a5]];
    var CYC = 9000;
    function draw(t) {
      var g = fitCanvas(canvas, 320, 190), tc = REDUCED ? 3000 : t % CYC;
      var gone = tc > 4200 && tc < 8400 ? Math.min(1, (tc - 4200) / 900) : 0;            /* seq 4 released, its leaves collapse */
      var leafFade = tc > 5200 && tc < 8400 ? ease((tc - 5200) / 900) : 0;
      var alive = function (n) { return (n === 'b1' || n === 'b') ? 1 - leafFade * (n === 'b1' ? 1 : (tc > 6400 ? ease((tc - 6400) / 900) : 0)) : 1; };
      g.lineCap = 'round';
      E.forEach(function (e) {
        var a = N[e[0]], b = N[e[1]], al = Math.min(alive(e[0]), alive(e[1]));
        g.strokeStyle = rgb(C.hair, al); g.lineWidth = 1.4; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
      });
      SEQ.forEach(function (s, i) {
        var grow = REDUCED ? 1 : ease((tc - i * 600) / 1100), al = i === 3 ? 1 - gone : 1;
        if (grow <= 0 || al <= 0) return;
        var pts = s[0].map(function (n) { return N[n]; }), segs = pts.length - 1, upto = grow * segs;
        g.strokeStyle = rgb(s[1], 0.55 * al); g.lineWidth = 4; g.lineJoin = 'round';
        g.beginPath(); g.moveTo(pts[0][0] + i - 1.5, pts[0][1] + i - 1.5);
        for (var j = 1; j <= segs; j++) {
          var f = Math.min(1, upto - (j - 1)); if (f <= 0) break;
          var a = pts[j - 1], b = pts[j];
          g.lineTo(a[0] + (b[0] - a[0]) * f + i - 1.5, a[1] + (b[1] - a[1]) * f + i - 1.5);
        }
        g.stroke();
      });
      Object.keys(N).forEach(function (k) {
        var p = N[k], al = alive(k); if (al <= 0.01) return;
        var r = k === 'r' ? 6 : 9 * (0.5 + 0.5 * al);
        if (k === 'r') { g.fillStyle = rgb(C.raised, 1); g.strokeStyle = rgb(C.ink, 0.9); g.lineWidth = 1.4; g.beginPath(); g.arc(p[0], p[1], r, 0, TAU); g.fill(); g.stroke(); return; }
        g.fillStyle = rgb(C.raised, al); g.strokeStyle = rgb((k === 'b1' || k === 'b') && gone > 0 ? C.c5 : C.ink, 0.85 * al); g.lineWidth = 1.4;
        g.beginPath(); g.roundRect ? g.roundRect(p[0] - r * 1.5, p[1] - r * 0.9, r * 3, r * 1.8, 5) : g.rect(p[0] - r * 1.5, p[1] - r * 0.9, r * 3, r * 1.8);
        g.fill(); g.stroke();
      });
      var msg = tc < 4200 ? ['4 sequences share their common prefixes', 'var(--ink)'] : tc < 6400 ? ['one sequence ends: its leaf is a free face', 'var(--coral-700)'] : tc < 8400 ? ['collapsed, and its parent is now free too', 'var(--coral-700)'] : ['4 sequences share their common prefixes', 'var(--ink)'];
      verdict(svg, msg[0], msg[1]);
    }
    register(canvas, draw, 3000);
  }

  function pillarTopo(canvas) {
    var C = palette(), I = icosphere(C), CENT = cubeCentroids();
    var cellCol = [C.b5, C.v5, C.m5, C.a5, mix(C.b5, C.raised, 0.35), mix(C.v5, C.raised, 0.35), mix(C.m5, C.raised, 0.35), mix(C.a5, C.raised, 0.35)];
    var seed = 7, rnd = function () { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
    var PTS = [];
    for (var i = 0; i < 120; i++) {
      var z = 2 * rnd() - 1, ph = TAU * rnd(), s = Math.sqrt(1 - z * z), p = [s * Math.cos(ph), s * Math.sin(ph), z], best = 0, bd = -2;
      for (var c = 0; c < 8; c++) { var d = p[0] * CENT[c][0] + p[1] * CENT[c][1] + p[2] * CENT[c][2]; if (d > bd) { bd = d; best = c; } }
      PTS.push({ p: p, cell: best });                 /* nearest by great-circle distance = largest dot */
    }
    var spr = cellCol.map(function (c) { return glowSprite(c, C); });
    var CYC = 9000;
    function draw(t) {
      var g = fitCanvas(canvas, 320, 190), tc = REDUCED ? CYC - 1 : t % CYC;
      var S2 = paintSphere(g, C, I, 160, 104, 72, 0.55 + (REDUCED ? 0 : t) * TAU / 40000, 0.38, (REDUCED ? 0 : t) / 14000, 1, [C.m5, C.b5, C.v5]);
      var shown = Math.min(PTS.length, Math.floor(tc / 55));
      S2.back();
      var put = function (front) {
        for (var k = 0; k < shown; k++) {
          var v = S2.view(PTS[k].p); if ((v[2] >= 0) !== front) continue;
          var age = Math.min(1, (tc - k * 55) / 400);
          g.globalAlpha = (front ? 1 : 0.3) * age;
          g.drawImage(spr[PTS[k].cell], v[0] - 4.5, v[1] - 4.5, 9, 9);
          g.globalAlpha = 1;
        }
        CENT.forEach(function (cc) {
          var v = S2.view(cc); if ((v[2] >= 0) !== front) return;
          g.strokeStyle = rgb(C.ink, front ? 0.6 : 0.2); g.lineWidth = 1.2; g.beginPath(); g.arc(v[0], v[1], 5, 0, TAU); g.stroke();
        });
      };
      put(false); S2.front(); put(true);
    }
    register(canvas, draw, CYC - 1);
  }

  function pillarCert(canvas) {
    var svg = canvas.parentNode.querySelector('svg'), C = palette();
    var P = [[0.06], [0.064], [0.07], [0.075], [0.31], [0.316], [0.322], [0.62], [0.627], [0.95], [0.956], [0.962]];
    var E = mst(P), X0 = 20, W = 280, L0 = -2.6, L1 = 0.3;
    var lx = function (h) { return X0 + (Math.log10(h) - L0) / (L1 - L0) * W; };
    var px = function (v) { return 20 + v[0] * 290; };
    var K = [[0, 0.02], [1500, 0.02], [3300, 0.1], [4800, 0.1], [6600, 0.9], [8100, 0.9], [9900, 0.02]];
    function draw(t) {
      var g = fitCanvas(canvas, 320, 190), s = REDUCED ? 0.1 : keyed(K, t % 9900, true), c = certify(P.length, E, s, 10), w = c.witness;
      g.lineCap = 'round';
      E.forEach(function (e) {
        var h = e[2], col = h < c.lo ? C.b5 : h <= c.hi ? C.c5 : C.g5, on = e === w;
        g.strokeStyle = rgb(col, h > c.hi ? 0.3 : 0.9); g.lineWidth = on ? 3.4 : 2;
        g.beginPath(); g.moveTo(px(P[e[0]]), 70); g.lineTo(px(P[e[1]]), 70); g.stroke();
      });
      P.forEach(function (p) { g.fillStyle = rgb(C.ink, 0.9); g.beginPath(); g.arc(px(p), 70, 3.2, 0, TAU); g.fill(); });
      if (w) [w[0], w[1]].forEach(function (i) { g.strokeStyle = rgb(C.c5, 1); g.lineWidth = 1.6; g.beginPath(); g.arc(px(P[i]), 70, 8, 0, TAU); g.stroke(); });
      var AY = 142, xl = Math.max(X0, lx(c.lo)), xh = Math.min(X0 + W, lx(c.hi));
      g.fillStyle = rgb(w ? C.c5 : C.hair, w ? 0.16 : 0.45); g.fillRect(xl, AY - 20, Math.max(0, xh - xl), 28);
      g.strokeStyle = rgb(C.hair, 1); g.lineWidth = 1; g.beginPath(); g.moveTo(X0, AY + 8); g.lineTo(X0 + W, AY + 8); g.stroke();
      g.strokeStyle = rgb(C.ink, 0.55); g.setLineDash([2, 3]); g.beginPath(); g.moveTo(lx(s), AY - 24); g.lineTo(lx(s), AY + 8); g.stroke(); g.setLineDash([]);
      E.forEach(function (e) {
        var h = e[2], col = h < c.lo ? C.b5 : h <= c.hi ? C.c5 : C.g5, x = lx(h);
        g.strokeStyle = rgb(col, 0.9); g.lineWidth = e === w ? 2.6 : 1.5; g.beginPath(); g.moveTo(x, AY + 6); g.lineTo(x, e === w ? AY - 18 : AY - 10); g.stroke();
      });
      if (w) verdict(svg, 'a merge height is in the band: refused, pair named', 'var(--coral-700)');
      else verdict(svg, 'β₀ = ' + c.count + ', certified', 'var(--green-700)');
    }
    register(canvas, draw, 0);
  }

  /* ============================================ painting and arrival */
  function onScreen(el) {
    var r = el.getBoundingClientRect(), vh = window.innerHeight || 800;
    return r.bottom > -80 && r.top < vh + 80 && r.width > 0;
  }
  function arrived(el) {
    var r = el.getBoundingClientRect(), vh = window.innerHeight || 800;
    return r.top < vh * 0.88 && r.bottom > vh * 0.08;
  }
  function paint(t) {
    for (var i = 0; i < FIGS.length; i++) {
      var f = FIGS[i];
      if (!onScreen(f.el)) { f.born = null; continue; }
      if (f.born == null) { if (!arrived(f.el)) continue; f.born = t; }
      try { f.draw(t - f.born); }
      catch (err) { if (!f.failed && window.console) console.error('figure failed', err); f.failed = true; }
    }
  }
  var reveals = [];
  function reveal() {
    var vh = window.innerHeight || 800;
    for (var i = reveals.length - 1; i >= 0; i--) {
      var el = reveals[i], r = el.getBoundingClientRect();
      if (r.top < vh * 0.92 && r.bottom > 0) {
        el.classList.add('is-in');
        (function (e) { setTimeout(function () { e.classList.add('is-done'); }, 1250); })(el);
        reveals.splice(i, 1);
      }
    }
  }

  function boot() {
    var el;
    if ((el = document.querySelector('canvas[data-fig="seal"]')) && el.getContext) figSeal(el);
    if ((el = document.querySelector('canvas[data-fig="p-stratum"]')) && el.getContext) pillarStratum(el);
    if ((el = document.querySelector('canvas[data-fig="p-foliation"]')) && el.getContext) pillarFoliation(el);
    if ((el = document.querySelector('canvas[data-fig="p-topo"]')) && el.getContext) pillarTopo(el);
    if ((el = document.querySelector('canvas[data-fig="p-cert"]')) && el.getContext) pillarCert(el);
    if ((el = document.querySelector('canvas[data-fig="t1"]')) && el.getContext) figT1(el);
    if ((el = document.querySelector('svg[data-fig="locate"]'))) figLocate(el);
    if ((el = document.getElementById('play-beta'))) figBeta(el);
    if ((el = document.getElementById('play-topk'))) figTopK(el);
    if ((el = document.getElementById('play-stratum'))) figStratum(el);
    if ((el = document.querySelector('svg[data-fig="foliation"]'))) figFoliation(el);
    if ((el = document.getElementById('play-gov'))) figGovernor(el);
    if ((el = document.querySelector('svg[data-fig="cells"]'))) figCells(el);
    if ((el = document.querySelector('pre[data-fig="boot"]'))) figBoot(el);

    if (REDUCED) {
      var still = function () { FIGS.forEach(function (f) { try { f.draw(f.end); } catch (err) { if (window.console) console.error(err); } }); };
      still();
      window.addEventListener('resize', still);
      return;
    }
    document.documentElement.classList.add('js-reveal');
    reveals = Array.prototype.slice.call(document.querySelectorAll('[data-reveal], .fsec__fig'));
    reveal();
    window.addEventListener('scroll', reveal, { passive: true });
    window.addEventListener('resize', reveal);
    /* every figure paints its first frame now, so none is ever blank */
    FIGS.forEach(function (f) { try { f.draw(0); } catch (err) { if (window.console) console.error(err); } });
    var ticked = false;
    var frame = function (t) { ticked = true; paint(t); requestAnimationFrame(frame); };
    requestAnimationFrame(frame);
    /* rAF can be throttled or never fire; the figures then run on a timer */
    setTimeout(function () {
      if (ticked) return;
      var t0 = 0;
      setInterval(function () { t0 += 50; paint(t0); reveal(); }, 50);
    }, 1200);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
