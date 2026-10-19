
const seedSelect = document.getElementById("seedSelect");
const methodSelect = document.getElementById("methodSelect");
const sparsitySelect = document.getElementById("sparsitySelect");

const testAccuracy = document.getElementById("testAccuracy");
const achievedSparsity = document.getElementById("achievedSparsity");
const activeWeights = document.getElementById("activeWeights");
const removedWeights = document.getElementById("removedWeights");
const statusPill = document.getElementById("statusPill");
const networkCanvas = document.getElementById("networkCanvas");

const SVG_NS = "http://www.w3.org/2000/svg";

let experiments = [];
let currentExperiment = null;
let currentEpoch = 200;
let replayTimer = null;

// ============================================================
// 1. SHARED HELPERS
// ============================================================

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tag);

  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }

  return element;
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function findExperiment(seed, method, target) {
  const actualMethod = Number(target) === 0
    ? "magnitude"
    : method;

  return experiments.find(experiment => {
    const summary = experiment.summary;

    return (
      Number(summary.seed) === Number(seed) &&
      summary.method === actualMethod &&
      Math.abs(
        Number(summary.target_sparsity) - Number(target)
      ) < 1e-9
    );
  });
}

function selectedExperiment() {
  return findExperiment(
    seedSelect.value,
    methodSelect.value,
    sparsitySelect.value
  );
}

// ============================================================
// 2. FINAL CHECKPOINT NETWORK VISUALIZATION
// ============================================================

function drawConnection(svg, x1, y1, x2, y2, active, weight) {
  const strength = Math.min(Math.abs(weight), 3) / 3;

  svg.appendChild(svgElement("line", {
    x1,
    y1,
    x2,
    y2,
    stroke: active ? "#43e7ff" : "#52637b",
    "stroke-width": active ? 1.3 + strength * 1.5 : 0.65,
    "stroke-opacity": active ? 0.45 + strength * 0.4 : 0.09,
    "stroke-linecap": "round"
  }));
}

function drawNeuron(svg, x, y, color, radius = 5) {
  svg.appendChild(svgElement("circle", {
    cx: x,
    cy: y,
    r: radius + 5,
    fill: color,
    "fill-opacity": 0.09
  }));

  svg.appendChild(svgElement("circle", {
    cx: x,
    cy: y,
    r: radius,
    fill: color,
    stroke: "#d6f5ff",
    "stroke-opacity": 0.45,
    "stroke-width": 0.6
  }));
}

function drawNetwork(experiment) {
  const { w1, w2, mask1, mask2 } = experiment.network;

  const svg = svgElement("svg", {
    viewBox: "0 0 760 430",
    width: "100%",
    height: "100%",
    role: "img",
    "aria-label":
      "Final checkpoint connectivity, showing active and pruned weights"
  });

  const inputX = 95;
  const hiddenX = 380;
  const outputX = 665;

  const inputY = [150, 280];
  const outputY = [150, 280];

  const hiddenY = Array.from(
    { length: 32 },
    (_, index) => 25 + index * (380 / 31)
  );

  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 32; j++) {
      drawConnection(
        svg,
        inputX,
        inputY[i],
        hiddenX,
        hiddenY[j],
        mask1[i][j],
        w1[i][j]
      );
    }
  }

  for (let i = 0; i < 32; i++) {
    for (let j = 0; j < 2; j++) {
      drawConnection(
        svg,
        hiddenX,
        hiddenY[i],
        outputX,
        outputY[j],
        mask2[i][j],
        w2[i][j]
      );
    }
  }

  for (const y of inputY) {
    drawNeuron(svg, inputX, y, "#43e7ff", 8);
  }

  for (const y of hiddenY) {
    drawNeuron(svg, hiddenX, y, "#ad7bff", 4);
  }

  for (const y of outputY) {
    drawNeuron(svg, outputX, y, "#3ce39a", 8);
  }

  for (const [label, x] of [
    ["INPUT", inputX],
    ["HIDDEN", hiddenX],
    ["OUTPUT", outputX]
  ]) {
    const text = svgElement("text", {
      x,
      y: 425,
      fill: "#97a8c3",
      "font-size": 12,
      "text-anchor": "middle",
      "letter-spacing": 2
    });

    text.textContent = label;
    svg.appendChild(text);
  }

  networkCanvas.replaceChildren(svg);
}

// ============================================================
// 3. DASHBOARD STYLING
// ============================================================

function installDashboardStyles() {
  const style = document.createElement("style");

  style.textContent = `
    .aqua-replay {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .aqua-replay-controls {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }

    .aqua-play-button {
      border: 1px solid rgba(67,231,255,0.4);
      background: rgba(67,231,255,0.12);
      color: #43e7ff;
      border-radius: 12px;
      padding: 11px 18px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }

    .aqua-play-button:hover {
      background: rgba(67,231,255,0.22);
    }

    .aqua-epoch-label {
      color: #e9f1ff;
      font-size: 14px;
      font-weight: 600;
    }

    .aqua-epoch-slider {
      width: 100%;
      accent-color: #43e7ff;
      cursor: pointer;
    }

    .aqua-replay-metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0,1fr));
      gap: 10px;
    }

    .aqua-replay-metric {
      background: rgba(15,29,50,0.9);
      border: 1px solid rgba(120,170,255,0.12);
      border-radius: 12px;
      padding: 12px;
      min-width: 0;
    }

    .aqua-replay-metric span {
      display: block;
      color: #97a8c3;
      font-size: 12px;
      margin-bottom: 6px;
    }

    .aqua-replay-metric strong {
      color: #e9f1ff;
      font-size: 18px;
    }

    .aqua-chart-block {
      background: rgba(7,14,26,0.65);
      border: 1px solid rgba(120,170,255,0.1);
      border-radius: 14px;
      padding: 12px;
    }

    .aqua-chart-title {
      color: #dce9ff;
      font-size: 13px;
      margin-bottom: 8px;
      font-weight: 600;
    }

    .aqua-chart-svg {
      width: 100%;
      height: auto;
      display: block;
    }

    .aqua-replay-note,
    .aqua-comparison-note {
      color: #97a8c3;
      font-size: 12px;
      line-height: 1.6;
    }

    .aqua-comparison {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .aqua-comparison-kicker {
      color: #43e7ff;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.7px;
    }

    .aqua-comparison-heading {
      color: #e9f1ff;
      font-size: 16px;
      font-weight: 700;
      margin-top: 5px;
    }

    .aqua-method-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0,1fr));
      gap: 12px;
    }

    .aqua-method-card {
      padding: 15px;
      border-radius: 16px;
      background: rgba(15,29,50,0.9);
      border: 1px solid rgba(120,170,255,0.14);
      min-width: 0;
    }

    .aqua-method-card.magnitude {
      border-top: 2px solid #43e7ff;
    }

    .aqua-method-card.taylor {
      border-top: 2px solid #ad7bff;
    }

    .aqua-method-label {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.1px;
      text-transform: uppercase;
      color: #97a8c3;
    }

    .aqua-method-value {
      display: block;
      margin: 12px 0;
      font-size: clamp(21px, 2vw, 30px);
      font-weight: 800;
      color: #e9f1ff;
      letter-spacing: -0.5px;
    }

    .aqua-method-sub {
      color: #97a8c3;
      font-size: 12px;
      line-height: 1.6;
    }

    .aqua-comparison-delta {
      padding: 14px 16px;
      background: rgba(67,231,255,0.07);
      border: 1px solid rgba(67,231,255,0.14);
      border-radius: 14px;
      color: #e9f1ff;
      font-size: 13px;
      line-height: 1.6;
    }

    .aqua-seed-title {
      font-size: 13px;
      font-weight: 700;
      color: #dce9ff;
    }

    .aqua-seed-row {
      margin-top: 13px;
    }

    .aqua-seed-labels {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: #97a8c3;
      font-size: 12px;
      margin-bottom: 7px;
    }

    .aqua-seed-bars {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .aqua-bar-track {
      width: 100%;
      height: 8px;
      border-radius: 10px;
      overflow: hidden;
      background: rgba(120,170,255,0.09);
    }

    .aqua-bar-fill {
      height: 100%;
      border-radius: 10px;
      transition: width 250ms ease;
    }

    .aqua-bar-fill.magnitude {
      background: #43e7ff;
    }

    .aqua-bar-fill.taylor {
      background: #ad7bff;
    }

    .aqua-comparison-summary {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      padding: 14px;
      border-radius: 14px;
      background: rgba(15,29,50,0.8);
      border: 1px solid rgba(120,170,255,0.12);
    }

    .aqua-summary-stat span {
      display: block;
      color: #97a8c3;
      font-size: 11px;
      margin-bottom: 7px;
    }

    .aqua-summary-stat strong {
      color: #e9f1ff;
      font-size: 17px;
    }

    .aqua-method-legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      color: #97a8c3;
      font-size: 12px;
    }

    .aqua-legend-key {
      display: inline-block;
      height: 8px;
      width: 17px;
      margin-right: 6px;
      border-radius: 4px;
    }

    @media (max-width: 600px) {
      .aqua-replay-metrics {
        grid-template-columns: 1fr;
      }

      .aqua-method-grid {
        grid-template-columns: 1fr;
      }
    }
  `;

  document.head.appendChild(style);
}

// ============================================================
// 4. TRAINING REPLAY
// ============================================================

function createReplayInterface() {
  const chartCard = document.querySelector(".chart-card");

  if (!chartCard) {
    throw new Error("Training Replay card not found.");
  }

  const placeholder = chartCard.querySelector(".chart-placeholder");

  if (placeholder) placeholder.remove();

  const replay = document.createElement("div");
  replay.className = "aqua-replay";

  replay.innerHTML = `
    <div class="aqua-replay-controls">
      <button class="aqua-play-button" id="aquaPlayButton">
        ▶ Play Replay
      </button>
      <span class="aqua-epoch-label" id="aquaEpochLabel">
        Epoch 200 / 200
      </span>
    </div>

    <input
      type="range"
      id="aquaEpochSlider"
      class="aqua-epoch-slider"
      min="1"
      max="200"
      value="200"
      step="1"
      aria-label="Select training epoch"
    >

    <div class="aqua-replay-metrics">
      <div class="aqua-replay-metric">
        <span>Training Loss</span>
        <strong id="aquaTrainLoss">--</strong>
      </div>

      <div class="aqua-replay-metric">
        <span>Validation Accuracy</span>
        <strong id="aquaValAccuracy">--</strong>
      </div>

      <div class="aqua-replay-metric">
        <span>Recorded Sparsity</span>
        <strong id="aquaEpochSparsity">--</strong>
      </div>
    </div>

    <div id="aquaLossChart"></div>
    <div id="aquaAccuracyChart"></div>
    <div id="aquaSparsityChart"></div>

    <p class="aqua-replay-note">
      Replay uses real per-epoch training records.
      The network above shows the final saved checkpoint,
      not reconstructed historical connection masks.
      Final test accuracy in Experiment Snapshot is not
      an epoch-by-epoch test measurement.
    </p>
  `;

  chartCard.appendChild(replay);

  document.getElementById("aquaPlayButton")
    .addEventListener("click", toggleReplay);

  document.getElementById("aquaEpochSlider")
    .addEventListener("input", event => {
      stopReplay();
      setEpoch(Number(event.target.value));
    });
}

function drawChart(containerId, title, values, color, maxY = null) {
  const container = document.getElementById(containerId);

  if (!container || !values.length) return;

  const width = 640;
  const height = 155;

  const left = 42;
  const right = 12;
  const top = 12;
  const bottom = 27;

  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const actualMax = Math.max(...values);
  const yMaximum = Math.max(maxY ?? actualMax * 1.1, 0.000001);

  const xPosition = index =>
    left + (index / Math.max(values.length - 1, 1)) * plotWidth;

  const yPosition = value =>
    top + plotHeight - (value / yMaximum) * plotHeight;

  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "aqua-chart-svg",
    role: "img",
    "aria-label": `${title} over recorded training epochs`
  });

  for (let i = 0; i <= 4; i++) {
    const y = top + (i / 4) * plotHeight;

    svg.appendChild(svgElement("line", {
      x1: left,
      y1: y,
      x2: width - right,
      y2: y,
      stroke: "#52637b",
      "stroke-opacity": 0.25,
      "stroke-width": 1
    }));

    const tick = svgElement("text", {
      x: left - 6,
      y: y + 4,
      fill: "#97a8c3",
      "font-size": 10,
      "text-anchor": "end"
    });

    tick.textContent = (yMaximum * (1 - i / 4)).toFixed(2);
    svg.appendChild(tick);
  }

  for (const epoch of [1, 50, 100, 150, 200]) {
    const tick = svgElement("text", {
      x: xPosition(epoch - 1),
      y: height - 6,
      fill: "#97a8c3",
      "font-size": 10,
      "text-anchor": "middle"
    });

    tick.textContent = epoch;
    svg.appendChild(tick);
  }

  const points = values.map((value, index) => [
    xPosition(index),
    yPosition(value)
  ]);

  svg.appendChild(svgElement("polyline", {
    points: points.map(([x, y]) => `${x},${y}`).join(" "),
    fill: "none",
    stroke: color,
    "stroke-opacity": 0.35,
    "stroke-width": 1.6,
    "stroke-linejoin": "round",
    "stroke-linecap": "round"
  }));

  const played = points.slice(0, currentEpoch);

  svg.appendChild(svgElement("polyline", {
    points: played.map(([x, y]) => `${x},${y}`).join(" "),
    fill: "none",
    stroke: color,
    "stroke-width": 2.5,
    "stroke-linejoin": "round",
    "stroke-linecap": "round"
  }));

  const currentIndex = Math.min(
    currentEpoch - 1,
    values.length - 1
  );

  svg.appendChild(svgElement("circle", {
    cx: points[currentIndex][0],
    cy: points[currentIndex][1],
    r: 4,
    fill: color,
    stroke: "#e9f1ff",
    "stroke-width": 1
  }));

  const block = document.createElement("div");
  block.className = "aqua-chart-block";

  const heading = document.createElement("div");
  heading.className = "aqua-chart-title";
  heading.textContent = title;

  block.appendChild(heading);
  block.appendChild(svg);

  container.replaceChildren(block);
}

function updateReplayCharts() {
  if (!currentExperiment) return;

  const history = currentExperiment.history;

  drawChart(
    "aquaLossChart",
    "Training Loss",
    history.map(row => row.train_loss),
    "#43e7ff"
  );

  drawChart(
    "aquaAccuracyChart",
    "Validation Accuracy",
    history.map(row => row.val_accuracy),
    "#3ce39a",
    1
  );

  drawChart(
    "aquaSparsityChart",
    "Network Sparsity",
    history.map(row => row.sparsity),
    "#ad7bff",
    1
  );
}

function setEpoch(epoch) {
  if (!currentExperiment) return;

  const history = currentExperiment.history;

  currentEpoch = Math.min(
    Math.max(Math.round(epoch), 1),
    history.length
  );

  const row = history[currentEpoch - 1];

  document.getElementById("aquaEpochSlider").value = currentEpoch;

  document.getElementById("aquaEpochLabel").textContent =
    `Epoch ${currentEpoch} / ${history.length}`;

  document.getElementById("aquaTrainLoss").textContent =
    Number(row.train_loss).toFixed(5);

  document.getElementById("aquaValAccuracy").textContent =
    percent(row.val_accuracy);

  document.getElementById("aquaEpochSparsity").textContent =
    percent(row.sparsity);

  updateReplayCharts();
}

function stopReplay() {
  if (replayTimer !== null) {
    clearInterval(replayTimer);
    replayTimer = null;
  }

  const button = document.getElementById("aquaPlayButton");

  if (button) button.textContent = "▶ Play Replay";
}

function toggleReplay() {
  if (!currentExperiment) return;

  if (replayTimer !== null) {
    stopReplay();
    return;
  }

  if (currentEpoch >= currentExperiment.history.length) {
    setEpoch(1);
  }

  document.getElementById("aquaPlayButton").textContent =
    "⏸ Pause Replay";

  replayTimer = setInterval(() => {
    if (currentEpoch >= currentExperiment.history.length) {
      stopReplay();
      return;
    }

    setEpoch(currentEpoch + 1);
  }, 90);
}

// ============================================================
// 5. MAGNITUDE VS TAYLOR COMPARISON
// ============================================================

function createComparisonInterface() {
  const card = document.querySelector(".comparison-card");

  if (!card) {
    throw new Error("Method Comparison card not found.");
  }

  const placeholder = card.querySelector(".comparison-placeholder");

  if (placeholder) placeholder.remove();

  const comparison = document.createElement("div");
  comparison.id = "aquaComparison";
  comparison.className = "aqua-comparison";

  card.appendChild(comparison);
}

function methodCard(method, experiment) {
  const s = experiment.summary;

  const card = document.createElement("div");

  card.className =
    `aqua-method-card ${method}`;

  const label = document.createElement("div");
  label.className = "aqua-method-label";
  label.textContent = `${method} pruning`;

  const value = document.createElement("strong");
  value.className = "aqua-method-value";
  value.textContent = percent(s.test_accuracy);

  const sub = document.createElement("div");
  sub.className = "aqua-method-sub";

  const accuracyLabel = document.createElement("div");
  accuracyLabel.textContent = "Final test accuracy";

  const weightLabel = document.createElement("div");
  weightLabel.textContent =
    `${s.active_weights} / ${s.total_weights} active weights`;

  const sparsityLabel = document.createElement("div");
  sparsityLabel.textContent =
    `${percent(s.achieved_sparsity)} achieved sparsity`;

  sub.append(
    accuracyLabel,
    weightLabel,
    sparsityLabel
  );

  card.append(label, value, sub);

  return card;
}

function createBarTrack(value, method) {
  const track = document.createElement("div");
  track.className = "aqua-bar-track";

  const fill = document.createElement("div");
  fill.className = `aqua-bar-fill ${method}`;

  fill.style.width =
    `${Math.max(0, Math.min(100, value * 100))}%`;

  track.appendChild(fill);

  return track;
}

function renderComparison() {
  const container = document.getElementById("aquaComparison");

  if (!container) return;

  container.replaceChildren();

  const seed = Number(seedSelect.value);
  const target = Number(sparsitySelect.value);
  const targetLabel = `${Math.round(target * 100)}%`;

  // There is only one dense reference run per seed.
  // It would be misleading to count it as two independently
  // trained pruning-method experiments.
  if (target === 0) {
    const dense = findExperiment(seed, "magnitude", 0);

    const heading = document.createElement("div");
    heading.className = "aqua-comparison-heading";
    heading.textContent = "Dense Baseline";

    const note = document.createElement("p");
    note.className = "aqua-comparison-note";
    note.textContent =
      "AQUA records one 0% dense baseline per seed. " +
      "Magnitude and Taylor pruning are compared only " +
      "at nonzero sparsity targets.";

    const stat = document.createElement("div");
    stat.className = "aqua-comparison-summary";

    if (dense) {
      stat.textContent =
        `Seed ${seed} • ${percent(dense.summary.test_accuracy)} ` +
        `test accuracy • ${dense.summary.active_weights} active weights`;
    }

    container.append(heading, stat, note);
    return;
  }

  const magnitude = findExperiment(seed, "magnitude", target);
  const taylor = findExperiment(seed, "taylor", target);

  if (!magnitude || !taylor) {
    container.textContent =
      "Recorded comparison data is unavailable.";
    return;
  }

  const intro = document.createElement("div");

  const kicker = document.createElement("div");
  kicker.className = "aqua-comparison-kicker";
  kicker.textContent = `Seed ${seed} • ${targetLabel} target`;

  const heading = document.createElement("div");
  heading.className = "aqua-comparison-heading";
  heading.textContent = "Same budget. Different choices.";

  intro.append(kicker, heading);

  // Selected-seed comparison.
  const methodGrid = document.createElement("div");
  methodGrid.className = "aqua-method-grid";

  methodGrid.append(
    methodCard("magnitude", magnitude),
    methodCard("taylor", taylor)
  );

  // Absolute difference in test accuracy percentage points.
  const difference =
    (taylor.summary.test_accuracy -
      magnitude.summary.test_accuracy) * 100;

  const delta = document.createElement("div");
  delta.className = "aqua-comparison-delta";

  const deltaLabel = document.createElement("strong");

  if (Math.abs(difference) < 1e-9) {
    deltaLabel.textContent =
      "Both methods achieved the same test accuracy.";
  } else {
    const leadingMethod =
      difference > 0 ? "Taylor" : "Magnitude";

    deltaLabel.textContent =
      `${leadingMethod} leads by ` +
      `${Math.abs(difference).toFixed(2)} percentage points.`;
  }

  const deltaDetail = document.createElement("div");
  deltaDetail.style.color = "#97a8c3";
  deltaDetail.style.marginTop = "5px";
  deltaDetail.textContent =
    "Comparison for the selected seed and sparsity target.";

  delta.append(deltaLabel, deltaDetail);

  // Cross-seed evidence.
  const title = document.createElement("div");
  title.className = "aqua-seed-title";
  title.textContent = "Test accuracy across all three seeds";

  const legend = document.createElement("div");
  legend.className = "aqua-method-legend";

  legend.innerHTML = `
    <span>
      <span class="aqua-legend-key"
            style="background:#43e7ff"></span>
      Magnitude
    </span>
    <span>
      <span class="aqua-legend-key"
            style="background:#ad7bff"></span>
      Taylor
    </span>
  `;

  const bars = document.createElement("div");

  const magnitudeScores = [];
  const taylorScores = [];

  for (const currentSeed of [42, 123, 2026]) {
    const mag = findExperiment(
      currentSeed, "magnitude", target
    );

    const tay = findExperiment(
      currentSeed, "taylor", target
    );

    if (!mag || !tay) continue;

    const magScore = Number(mag.summary.test_accuracy);
    const tayScore = Number(tay.summary.test_accuracy);

    magnitudeScores.push(magScore);
    taylorScores.push(tayScore);

    const row = document.createElement("div");
    row.className = "aqua-seed-row";

    const labels = document.createElement("div");
    labels.className = "aqua-seed-labels";

    const seedLabel = document.createElement("span");
    seedLabel.textContent = `Seed ${currentSeed}`;

    const scoreLabel = document.createElement("span");
    scoreLabel.textContent =
      `M ${percent(magScore)}  ·  T ${percent(tayScore)}`;

    labels.append(seedLabel, scoreLabel);

    const tracks = document.createElement("div");
    tracks.className = "aqua-seed-bars";

    tracks.append(
      createBarTrack(magScore, "magnitude"),
      createBarTrack(tayScore, "taylor")
    );

    row.append(labels, tracks);
    bars.appendChild(row);
  }

  const mean = values =>
    values.reduce((sum, value) => sum + value, 0) /
    values.length;

  const meanMagnitude = mean(magnitudeScores);
  const meanTaylor = mean(taylorScores);

  const summary = document.createElement("div");
  summary.className = "aqua-comparison-summary";

  for (const [label, value] of [
    ["Magnitude mean", percent(meanMagnitude)],
    ["Taylor mean", percent(meanTaylor)]
  ]) {
    const stat = document.createElement("div");
    stat.className = "aqua-summary-stat";

    const statLabel = document.createElement("span");
    statLabel.textContent = label;

    const statValue = document.createElement("strong");
    statValue.textContent = value;

    stat.append(statLabel, statValue);
    summary.appendChild(stat);
  }

  const note = document.createElement("p");
  note.className = "aqua-comparison-note";
  note.textContent =
    "All values come from recorded AQUA test results. " +
    "Means use three random seeds and are descriptive, " +
    "not proof of statistical significance. " +
    "Accuracy bars use a 0–100% scale.";

  container.append(
    intro,
    methodGrid,
    delta,
    title,
    legend,
    bars,
    summary,
    note
  );
}

// ============================================================
// 6. UPDATE EXPERIMENT SELECTION
// ============================================================

function updateDashboard() {
  stopReplay();

  const experiment = selectedExperiment();

  if (!experiment) {
    currentExperiment = null;
    statusPill.textContent = "Experiment not found";
    networkCanvas.textContent =
      "No recorded model available.";

    return;
  }

  currentExperiment = experiment;

  const summary = experiment.summary;

  testAccuracy.textContent =
    percent(summary.test_accuracy);

  achievedSparsity.textContent =
    percent(summary.achieved_sparsity);

  activeWeights.textContent =
    summary.active_weights;

  removedWeights.textContent =
    summary.removed_weights;

  const methodLabel =
    Number(summary.target_sparsity) === 0
      ? "Dense baseline"
      : `${summary.method} pruning`;

  statusPill.textContent =
    `Seed ${summary.seed} • ${methodLabel}`;

  drawNetwork(experiment);

  // Start each selected experiment at its final epoch.
  setEpoch(experiment.history.length);

  renderComparison();
}

// ============================================================
// 7. LOAD RECORDED AQUA DATA
// ============================================================

async function loadDashboard() {
  try {
    statusPill.textContent =
      "Loading recorded AQUA experiments...";

    const response = await fetch("data/aqua_results.json");

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (
      !Array.isArray(data.experiments) ||
      data.experiments.length !== 27
    ) {
      throw new Error("Expected 27 recorded experiments.");
    }

    for (const experiment of data.experiments) {
      if (
        !Array.isArray(experiment.history) ||
        experiment.history.length !== 200
      ) {
        throw new Error(
          `Invalid history for ${experiment.id}`
        );
      }
    }

    experiments = data.experiments;

    installDashboardStyles();
    createReplayInterface();
    createComparisonInterface();

    updateDashboard();

  } catch (error) {
    console.error(
      "AQUA Observatory loading error:",
      error
    );

    statusPill.textContent = "Data loading failed";

    networkCanvas.textContent =
      "Unable to load recorded AQUA data. " +
      "Check the browser console.";
  }
}

seedSelect.addEventListener(
  "change",
  updateDashboard
);

methodSelect.addEventListener(
  "change",
  updateDashboard
);

sparsitySelect.addEventListener(
  "change",
  updateDashboard
);

loadDashboard();
