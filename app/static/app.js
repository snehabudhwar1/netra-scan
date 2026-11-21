"use strict";

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const GRADES = [
  { name: "No DR", color: "var(--g0)" },
  { name: "Mild", color: "var(--g1)" },
  { name: "Moderate", color: "var(--g2)" },
  { name: "Severe", color: "var(--g3)" },
  { name: "Proliferative", color: "var(--g4)" },
];
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let chosen = null;
let objectUrl = null;
let images = null;
let busy = false;

// tiny helpers: el("div", {class: "x"}, ["text", node]) and the same for SVG
function el(tag, attrs = {}, kids = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "style") n.setAttribute("style", v);
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  (Array.isArray(kids) ? kids : [kids]).forEach((c) => {
    if (c === null || c === undefined || c === false) return;
    n.append(c.nodeType ? c : document.createTextNode(String(c)));
  });
  return n;
}
function sv(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}
const pct = (v) => (v === null || v === undefined ? "n/a" : `${Math.round(v * 100)}%`);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ------------------------------------------------------------------ the lens artwork
function drawDial() {
  const svg = $("dial");
  svg.append(sv("circle", { cx: 200, cy: 200, r: 198, fill: "none", stroke: "#4a2f3b", "stroke-width": 1.5 }));
  const g = sv("g", { class: "ticks" });
  for (let i = 0; i < 120; i++) {
    const a = (i * 3 * Math.PI) / 180;
    const long = i % 10 === 0, mid = i % 5 === 0;
    const r1 = long ? 183 : mid ? 188 : 192;
    g.append(sv("line", {
      x1: 200 + r1 * Math.cos(a), y1: 200 + r1 * Math.sin(a),
      x2: 200 + 198 * Math.cos(a), y2: 200 + 198 * Math.sin(a),
      stroke: long ? "#bba9b2" : "#5a3e4b", "stroke-width": long ? 1.6 : 1,
    }));
  }
  svg.append(g);
}

function rng(seed) {                       // small seeded random generator, so the picture is always the same
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function drawSample() {
  const rnd = rng(11);
  const vessels = $("vessels"), lesions = $("lesions");
  const cubic = (p, t) => {
    const u = 1 - t;
    return [
      u * u * u * p[0][0] + 3 * u * u * t * p[1][0] + 3 * u * t * t * p[2][0] + t * t * t * p[3][0],
      u * u * u * p[0][1] + 3 * u * u * t * p[1][1] + 3 * u * t * t * p[2][1] + t * t * t * p[3][1],
    ];
  };
  const tangent = (p, t) => {
    const a = cubic(p, clamp(t - 0.01, 0, 1)), b = cubic(p, clamp(t + 0.01, 0, 1));
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  };
  const add = (d, w) => vessels.append(sv("path", { d, "stroke-width": w.toFixed(2), opacity: 0.82 }));

  function branch(x, y, ang, len, w, depth) {
    const ex = x + len * Math.cos(ang), ey = y + len * Math.sin(ang);
    const bend = (rnd() - 0.5) * 0.7;
    const cx = x + (len * 0.55) * Math.cos(ang + bend), cy = y + (len * 0.55) * Math.sin(ang + bend);
    add(`M${x.toFixed(1)},${y.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`, w);
    if (depth > 0) {
      for (let k = 0; k < 2; k++) {
        const t = 0.4 + rnd() * 0.4, s = rnd() > 0.5 ? 1 : -1;
        const px = x + (ex - x) * t, py = y + (ey - y) * t;
        branch(px, py, ang + s * (0.5 + rnd() * 0.5), len * 0.5, w * 0.6, depth - 1);
      }
    }
  }

  // four main arcades leaving the optic disc
  const arcs = [
    [[140, 186], [195, 118], [300, 96], [376, 160], 5.2],
    [[140, 208], [195, 290], [300, 306], [376, 244], 5.6],
    [[116, 186], [100, 150], [90, 118], [62, 92], 4.4],
    [[116, 208], [100, 246], [90, 282], [62, 310], 4.6],
    [[142, 196], [210, 190], [290, 196], [350, 200], 2.6],
  ];
  arcs.forEach((a) => {
    const p = a.slice(0, 4), w = a[4];
    add(`M${p[0]} C${p[1]} ${p[2]} ${p[3]}`, w);
    const count = w > 3 ? 5 : 3;
    for (let k = 1; k <= count; k++) {
      const t = 0.18 + (k / (count + 1)) * 0.75 + (rnd() - 0.5) * 0.06;
      const [px, py] = cubic(p, t);
      const base = tangent(p, t), s = k % 2 ? 1 : -1;
      branch(px, py, base + s * (0.45 + rnd() * 0.5), 46 + rnd() * 50, w * 0.52, 2);
    }
  });

  // a few spots, like the ones the tool looks for
  for (let i = 0; i < 9; i++) {
    lesions.append(sv("circle", {
      cx: (288 + rnd() * 26).toFixed(1), cy: (240 + rnd() * 30).toFixed(1), r: (1.6 + rnd() * 2.6).toFixed(1),
      fill: "#F8E08A", opacity: 0.92,
    }));
  }
  [[238, 262], [300, 168], [212, 242], [262, 150]].forEach(([x, y]) => {
    lesions.append(sv("circle", { cx: x, cy: y, r: (3 + rnd() * 2.2).toFixed(1), fill: "#4c0f08", opacity: 0.8 }));
  });
}

drawDial();
drawSample();

// ------------------------------------------------------------------ choosing a photo
const lens = $("lens"), fileInput = $("file"), frame = $("frame");

function isTiff(f) { return f.type === "image/tiff" || /\.(tif|tiff)$/i.test(f.name); }

function setStatus(t) { $("status").textContent = t; }

function resetAll() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null; chosen = null; images = null; lastResult = null;
  fileInput.value = "";
  $("preview").hidden = true; $("viewer").hidden = true;
  $("sample").hidden = false; $("callouts").classList.remove("off");
  $("lens-msg").hidden = false; $("lens-msg").textContent = "Drop a retina photo here, or click to choose";
  $("views").hidden = true;
  $("go").disabled = true; $("reset").hidden = true;
  $("report").hidden = true; $("sheet").replaceChildren();
  setStatus("Illustration of a retina. Drop your own photo to begin.");
}

function setFile(f) {
  if (!f || busy) return;
  if (!f.type.startsWith("image/") && !/\.(tif|tiff)$/i.test(f.name)) {
    setStatus("That file is not an image. Choose a JPG, PNG or TIFF photo.");
    return;
  }
  if (f.size > 15 * 1024 * 1024) { setStatus("That file is larger than 15 MB. Choose a smaller photo."); return; }
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null; images = null;
  chosen = f;
  $("viewer").hidden = true; $("views").hidden = true;
  $("report").hidden = true; $("sheet").replaceChildren();
  $("sample").hidden = true; $("callouts").classList.add("off");
  if (isTiff(f)) {
    $("preview").hidden = true;
    $("lens-msg").hidden = false;
    $("lens-msg").textContent = `${f.name}: TIFF has no preview in the browser`;
  } else {
    objectUrl = URL.createObjectURL(f);
    $("preview").src = objectUrl; $("preview").hidden = false;
    $("lens-msg").hidden = true;
  }
  $("go").disabled = false; $("reset").hidden = false;
  setStatus(`${f.name} is ready. Select Analyse photo.`);
}

$("pick").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
lens.addEventListener("click", () => { if (!busy) fileInput.click(); });
lens.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !busy) { e.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach((t) => lens.addEventListener(t, (e) => { e.preventDefault(); lens.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => lens.addEventListener(t, (e) => { e.preventDefault(); lens.classList.remove("drag"); }));
lens.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
$("reset").addEventListener("click", resetAll);
$("go").addEventListener("click", analyse);

// the three views inside the lens
document.querySelectorAll("#views button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
function showView(name) {
  if (!images || !images[name]) return;
  $("viewer").src = images[name];
  $("viewer").alt = { retina: "Cleaned retina photo", enhanced: "Contrast-enhanced view", vessels: "Blood-vessel map" }[name];
  $("viewer").hidden = false; $("preview").hidden = true;
  document.querySelectorAll("#views button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === name)));
}

// ------------------------------------------------------------------ talking to the server
const STAGES = ["Checking the photo quality", "Reading blood vessels, spots and colour", "Estimating the grade"];
let stageTimer = null;

function startScan() {
  busy = true; frame.classList.add("scanning");
  $("go").disabled = true; $("pick").disabled = true; $("reset").disabled = true;
  let i = 0;
  setStatus(STAGES[0] + "...");
  stageTimer = setInterval(() => { i = Math.min(i + 1, STAGES.length - 1); setStatus(STAGES[i] + "..."); }, 1500);
}
function stopScan() {
  busy = false; frame.classList.remove("scanning");
  clearInterval(stageTimer);
  $("go").disabled = false; $("pick").disabled = false; $("reset").disabled = false;
}

async function analyse() {
  if (!chosen || busy) return;
  startScan();
  const form = new FormData();
  form.append("file", chosen);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `The server answered with an error (${res.status}).`);
    stopScan();
    render(body);
  } catch (e) {
    stopScan();
    showError(e.message || "The server could not be reached. Check that it is running, then try again.");
  }
}

function openReport() {
  $("report").hidden = false;
  $("sheet").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
  $("sheet").focus({ preventScroll: true });
}

function showError(msg) {
  setStatus("The photo could not be analysed.");
  $("sheet").replaceChildren(
    sheetHead("Screening report", ""),
    el("div", { class: "note bad", role: "alert" }, [
      el("h3", {}, "The photo could not be analysed"),
      el("p", {}, msg),
      el("p", {}, "Choose the photo again, or try a different one."),
    ]),
    el("div", { class: "row-btns no-print" }, [
      el("button", { class: "btn ink", type: "button", onclick: () => { resetAll(); $("top").scrollIntoView(); fileInput.click(); } }, "Choose another photo"),
    ]));
  openReport();
}

// ------------------------------------------------------------------ the report sheet
function sheetHead(title, meta) {
  const lines = Array.isArray(meta) ? meta : [meta];
  return el("div", { class: "sheet-head" }, [
    el("h2", { id: "report-h" }, title),
    el("p", { class: "meta" }, lines.filter(Boolean).map((t) => el("span", {}, t))),
  ]);
}

// ------------------------------------------------------------------ patient details (kept in this tab only)
let lastResult = null;

function reportId() {
  const d = new Date(), z = (n) => String(n).padStart(2, "0");
  const rand = Math.floor(Math.random() * 0xffff).toString(16).toUpperCase().padStart(4, "0");
  return `NS-${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}-${rand}`;
}

function readPatient() {
  const v = (id) => $(id).value.trim();
  return {
    name: v("pt-name"), age: v("pt-age"), sex: v("pt-sex"), type: v("pt-type"),
    years: v("pt-years"), eye: v("pt-eye"), notes: v("pt-notes"),
    symptoms: [...document.querySelectorAll('#pt-form input[name="sym"]:checked')].map((c) => c.value),
  };
}

// the patient card and the symptom note; rebuilt whenever the form changes
function patientNodes() {
  if (!lastResult) return [];
  const p = readPatient();
  const item = (label, value, cls = "", blank = false) =>
    el("div", { class: cls }, [el("dt", {}, label), el("dd", { class: blank && !value ? "blank" : "" }, value || (blank ? "" : "Not given"))]);
  const rows = [
    item("Name or ID", p.name, "w2", true),
    item("Age", p.age ? `${p.age} years` : "", "", true),
    item("Sex", p.sex, "", true),
    item("Diabetes type", p.type),
    item("Years with diabetes", p.years),
    item("Eye in the photo", p.eye, "", true),
  ];
  const symptomText = p.symptoms.length ? p.symptoms.join(", ") : "None ticked";
  rows.push(item("Symptoms", symptomText, "w2"));
  rows.push(item("Photo file", chosen ? chosen.name : "", "w2"));
  if (p.notes) rows.push(el("div", { class: "w4" }, [el("dt", {}, "Notes"), el("dd", { class: "long" }, p.notes)]));

  const out = [el("div", { class: "pt-card" }, [el("h3", {}, "Patient"), el("dl", { class: "pt-dl" }, rows)])];

  if (p.symptoms.includes("Sudden loss of vision")) {
    out.push(el("div", { class: "note bad", role: "alert" }, [
      el("h3", {}, "Sudden loss of vision needs urgent care"),
      el("p", {}, "Please see an eye doctor or go to an emergency department today, whatever this photo result says."),
    ]));
  } else if (p.symptoms.length && lastResult.status !== "rejected" && !lastResult.referable) {
    out.push(el("div", { class: "note", role: "note" }, [
      el("h3", {}, "Symptoms were reported"),
      el("p", {}, `${p.symptoms.join(", ")}. A low-risk photo result does not rule out an eye problem. Please see an eye doctor about these symptoms.`),
    ]));
  }
  return out;
}

function refreshPatient() {
  const slot = $("pt-slot");
  if (slot) slot.replaceChildren(...patientNodes());
}

function doctorBox() {
  return el("div", { class: "print-only" }, el("div", { class: "doc-box" }, [
    el("h3", {}, "Reviewed by an eye doctor"),
    el("div", { class: "doc-lines" }, [
      el("span", {}, "Doctor's name"), el("span", {}, "Signature"), el("span", {}, "Date"),
      el("span", { class: "notes" }, "Notes"),
    ]),
  ]));
}

$("pt-form").addEventListener("submit", (e) => e.preventDefault());
$("pt-form").addEventListener("input", refreshPatient);
$("pt-clear").addEventListener("click", () => { $("pt-form").reset(); refreshPatient(); $("pt-name").focus(); });
$("plink").addEventListener("click", () => setTimeout(() => $("pt-name").focus({ preventScroll: true }), 400));
$("pt-back").addEventListener("click", () => setTimeout(() => $("pick").focus({ preventScroll: true }), 400));

function qualityList(checks) {
  if (!checks || !checks.length) return null;
  const mark = { ok: "✓", warn: "!", fail: "×" };
  const word = { ok: "passed", warn: "borderline", fail: "failed" };
  return el("div", { class: "block" }, [
    el("h3", {}, "Photo quality"),
    el("ul", { class: "qc" }, checks.map((c) => el("li", {}, [
      el("span", { class: `ic ${c.status}`, role: "img", "aria-label": word[c.status] }, mark[c.status]),
      el("div", {}, [el("span", { class: "name" }, c.label), c.message ? el("span", { class: "why" }, c.message) : null]),
    ]))),
  ]);
}

function render(r) {
  const sheet = $("sheet");
  lastResult = r;
  const meta = [`Report ${reportId()}`, new Date().toLocaleString(), `Model ${r.model_version}`];
  const parts = [sheetHead("Screening report", meta), el("div", { class: "pt-slot", id: "pt-slot" }, patientNodes())];

  if (r.status === "rejected") {
    setStatus("This photo cannot be graded. See the report below.");
    parts.push(el("div", { class: "note bad", role: "alert" }, [
      el("h3", {}, "This photo was not graded"),
      el("p", {}, r.message || "The photo is not good enough to grade."),
      el("p", {}, "Retake the photo using the advice above, then upload it again."),
    ]));
    const q = qualityList(r.quality && r.quality.checks);
    if (q) parts.push(q);
    parts.push(el("p", { class: "fine" }, r.disclaimer));
    parts.push(el("div", { class: "row-btns no-print" }, [
      el("button", { class: "btn ink", type: "button", onclick: again }, "Choose another photo"),
    ]));
    sheet.replaceChildren(...parts);
    refreshPatient();
    openReport();
    return;
  }

  images = r.images || null;
  if (images) { $("views").hidden = false; showView("retina"); }
  setStatus(r.referable ? "Result ready. An eye-doctor visit is advised." : "Result ready. No urgent signs were found.");

  if (r.status === "uncertain") {
    parts.push(el("div", { class: "note", role: "alert" }, [
      el("h3", {}, "Uncertain result. Please have a doctor look at this photo."),
      el("ul", {}, (r.uncertain_reasons || []).map((t) => el("li", {}, t))),
    ]));
  }

  const g = GRADES[r.grade];
  const refPct = Math.round(r.referable_probability * 100);
  const thrPct = Math.round(r.refer_threshold * 100);
  const vc = r.referable ? "var(--g3)" : "var(--g0)";

  // left column: the grade
  const left = el("div", { class: "stack" }, [
    el("div", { class: "block" }, [
      el("div", { class: "grade-hero" }, [
        el("div", { class: "numeral", "aria-hidden": "true" }, String(r.grade)),
        el("div", {}, [el("h3", { class: "grade-name" }, `Grade ${r.grade}: ${g.name}`), el("p", { class: "grade-text" }, r.grade_text)]),
      ]),
      el("ol", { class: "scale", "aria-label": "Grade scale from 0 to 4" }, GRADES.map((x, i) =>
        el("li", { class: i === r.grade ? "on" : "", style: `--c:${x.color}`, "aria-current": i === r.grade ? "true" : "false" },
          el("span", {}, [String(i), " ", el("span", { class: "nm" }, x.name)])))),
      el("p", { class: "advice" }, r.advice),
    ]),
    el("div", { class: "block" }, [
      el("h3", {}, "How sure the model is about each grade"),
      el("div", { class: "probs" }, Object.entries(r.probabilities).map(([name, p], i) =>
        el("div", { class: `prob${i === r.grade ? " top" : ""}`, style: `--c:${GRADES[i].color}` }, [
          el("span", {}, `${i} ${name}`),
          el("div", { class: "track" }, el("i", { style: `width:${Math.max(Math.round(p * 100), 1)}%` })),
          el("span", { class: "val" }, `${Math.round(p * 100)}%`),
        ]))),
      el("p", { class: "conf" }, `Confidence in the top grade: ${Math.round(r.confidence * 100)}%. The exact grade is harder to name than the visit advice, so treat it as an estimate.`),
    ]),
  ]);

  // right column: the decision and the photo quality
  const right = el("div", { class: "stack" }, [
    el("div", { class: "block", style: `--vc:${vc}` }, [
      el("h3", {}, "Is an eye-doctor visit needed?"),
      el("p", { class: "verdict" }, r.referable ? "Visit advised" : "No urgent signs found"),
      el("p", { class: "big-pct" }, [`${refPct}%`, el("small", {}, "chance, by the model's estimate")]),
      el("div", { class: "meter", role: "img", "aria-label": `${refPct} percent. The tool advises a visit above ${thrPct} percent.` }, [
        el("div", { class: "fill", style: `width:${refPct}%` }),
        el("div", { class: "tick", style: `left:${thrPct}%` }),
        el("div", { class: "tick-label", style: `left:${clamp(thrPct, 16, 84)}%` }, `Visit advised above ${thrPct}%`),
      ]),
      el("p", { class: "meter-note" }, "The line is set low on purpose, so that few people who need a visit are missed. It also means some people without disease are advised to go."),
    ]),
    qualityList(r.quality && r.quality.checks),
  ]);

  parts.push(el("div", { class: "cols" }, [left, right]));

  if (r.images) {
    const fig = (src, cap) => el("figure", {}, [el("img", { src, alt: cap }), el("figcaption", {}, cap)]);
    parts.push(el("div", { class: "print-only" }, [
      el("div", { class: "pics" }, [fig(r.images.retina, "Cleaned photo"), fig(r.images.enhanced, "Contrast view"), fig(r.images.vessels, "Vessel map")]),
    ]));
  }

  parts.push(doctorBox());
  parts.push(el("p", { class: "fine" }, r.disclaimer));
  parts.push(el("div", { class: "row-btns no-print" }, [
    el("button", { class: "btn ink", type: "button", onclick: () => window.print() }, "Save report as PDF"),
    el("button", { class: "btn outline", type: "button", onclick: again }, "Analyse another photo"),
  ]));

  sheet.replaceChildren(...parts);
  openReport();
}

function again() {
  resetAll();
  $("top").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" });
  setTimeout(() => fileInput.click(), 350);
}

// ------------------------------------------------------------------ the honest numbers
const DATASET_NAMES = { idrid: "IDRiD", aptos: "APTOS 2019", messidor: "Messidor" };
const niceName = (d) => (d ? DATASET_NAMES[String(d).toLowerCase()] || d : "another dataset");

function metricCard(title, sub, rows, fmt) {
  return el("div", { class: "metric" }, [
    el("h3", {}, title), el("p", { class: "sub" }, sub),
    ...rows.filter((r) => r.value !== null && r.value !== undefined).map((r) =>
      el("div", { class: "bar-row", style: `--c:${r.color}` }, [
        el("span", {}, r.label),
        el("div", { class: "track" }, el("i", { style: `width:${clamp(r.value, 0, 1) * 100}%` })),
        el("b", {}, fmt(r.value)),
      ])),
  ]);
}

async function loadInfo() {
  const box = $("numbers-body");
  try {
    const res = await fetch("/api/model-info");
    if (!res.ok) throw new Error("not ready");
    const m = await res.json();
    const base = m.baseline_metadata_only, internal = m.internal_test, ext = m.external_test;
    const extName = ext ? niceName(ext.dataset) : null;

    const series = (key) => [
      { label: "Shortcut", color: "var(--s-base)", value: base ? base[key] : null },
      { label: "Same source", color: "var(--s-int)", value: internal ? internal[key] : null },
      { label: "New clinic", color: "var(--s-ext)", value: ext ? ext[key] : null },
    ];
    const num = (v) => v.toFixed(2);

    const kids = [
      el("p", { class: "intro" }, "These scores come from photos the model never saw while learning. The orange bars matter most: they are photos from a different clinic and camera, which is closer to real use."),
      el("div", { class: "legend" }, [
        el("span", { style: "--c:var(--s-base)" }, [el("i"), "Shortcut that never looks at the retina"]),
        el("span", { style: "--c:var(--s-int)" }, [el("i"), `Netra Scan, photos from the same source${internal ? ` (${internal.n})` : ""}`]),
        ext ? el("span", { style: "--c:var(--s-ext)" }, [el("i"), `Netra Scan, new clinic: ${extName} (${ext.n})`]) : null,
      ]),
      el("div", { class: "metrics" }, [
        metricCard("Grade score (QWK)", "From 0 to 1. Being three grades off costs more than being one grade off. Higher is better.", series("qwk"), num),
        metricCard("Finds people who need a visit", "Of everyone with moderate disease or worse, the share the tool advised to see a doctor.", series("referable_sensitivity"), pct),
        metricCard("Exact grade right", "The share of photos where the grade from 0 to 4 was named exactly.", series("accuracy"), pct),
        metricCard("Clears people who do not need a visit", "Of everyone with no or mild disease, the share the tool did not send to a doctor.", series("referable_specificity"), pct),
      ]),
    ];

    // how often each grade is named exactly
    const rc = (x) => (x && x.recall_per_grade && Object.keys(x.recall_per_grade).length ? x.recall_per_grade : null);
    if (rc(internal)) {
      const cols = GRADES.map((g, i) => {
        const a = rc(internal)[String(i)], b = rc(ext) ? rc(ext)[String(i)] : null;
        const bar = (v, color) => v === null || v === undefined ? null
          : el("i", { style: `height:${Math.max(v * 140, 2)}px;--c:${color}` }, el("b", {}, pct(v)));
        return el("div", { class: "col5" }, [
          el("div", { class: "pair" }, [bar(a, "var(--s-int)"), bar(b, "var(--s-ext)")]),
          el("span", {}, `${i} ${g.name}`),
        ]);
      });
      kids.push(el("div", { class: "recall" }, [
        el("h3", {}, "Which grades are named correctly"),
        el("p", { class: "sub" }, "For photos that really have each grade, the share the model named exactly. Mild disease and the most advanced grades are the hardest."),
        el("div", { class: "cols5" }, cols),
      ]));
    }

    if (internal && ext && internal.qwk !== null && ext.qwk !== null) {
      const dq = (internal.qwk - ext.qwk).toFixed(2);
      kids.push(el("div", { class: "finding" }, [
        el("h3", {}, "What the gap tells us"),
        el("p", {}, `On photos from ${extName}, the grade score fell by ${dq}, and the tool advised a visit for ${pct(ext.referable_sensitivity)} of the people who needed one, compared with ${pct(internal.referable_sensitivity)} on photos from the training source. A model like this should be checked on photos from each new clinic before anyone relies on it.`),
      ]));
    } else if (!ext) {
      kids.push(el("div", { class: "finding" }, [
        el("h3", {}, "New-clinic test not run yet"),
        el("p", {}, "Scores on photos from a different clinic are not available yet. They are usually lower than the scores above."),
      ]));
    }

    kids.push(el("p", { class: "how-measured" }, [
      el("strong", {}, "How this was measured. "),
      `The model is ${m.model} using ${m.features} features. It learned from ${niceName(m.trained_on)} photos. "Same source" is a part of those photos that was held back before learning started. `,
      "The shortcut is a tiny model that sees only image size and brightness, never the retina. It still scores well because different cameras and clinics leave a trace in the picture, which is why every real model is compared against it.",
    ]));

    box.replaceChildren(...kids);
  } catch (e) {
    box.replaceChildren(el("p", { class: "intro" }, "The test results are not available right now. Reload the page in a minute."));
  }
}
loadInfo();
