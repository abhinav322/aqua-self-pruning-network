
/*
 * AQUA Observatory — Inference Efficiency Lab
 * Displays real, recorded benchmark measurements.
 * No simulated performance values.
 */

(async function initializeEfficiencyLab() {
  const response = await fetch("data/benchmark_results.json");

  if (!response.ok) {
    console.error("Efficiency data unavailable:", response.status);
    return;
  }

  const dataset = await response.json();
  const rows = dataset.measurements;

  if (!Array.isArray(rows) || !rows.length) {
    console.error("No benchmark measurements found.");
    return;
  }

  const dashboard = document.querySelector(".dashboard-grid");

  if (!dashboard) {
    console.error("Dashboard grid not found.");
    return;
  }

  // Styles are scoped to this section.
  const styles = document.createElement("style");

  styles.textContent = `
    .efficiency-lab {
      grid-column: 2 / 4;
    }

    .efficiency-subtitle {
      color: #97a8c3;
      line-height: 1.6;
      font-size: 14px;
      margin-bottom: 20px;
    }

    .efficiency-controls {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 14px;
      margin-bottom: 22px;
    }

    .efficiency-metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }

    .efficiency-metric {
      background: rgba(15,29,50,0.9);
      border: 1px solid rgba(120,170,255,0.13);
      border-radius: 15px;
      padding: 16px;
    }

    .efficiency-metric span {
      color: #97a8c3;
      display: block;
      font-size: 12px;
      margin-bottom: 9px;
    }

    .efficiency-metric strong {
      color: #e9f1ff;
      font-size: 22px;
    }

    .efficiency-chart {
      border: 1px solid rgba(120,170,255,0.12);
      background: rgba(7,14,26,0.65);
      padding: 20px;
      border-radius: 16px;
      margin-bottom: 20px;
    }

    .efficiency-chart h3 {
      font-size: 15px;
      font-weight: 700;
      margin: 0 0 20px;
    }

    .efficiency-bar-label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: #97a8c3;
      font-size: 13px;
      margin-bottom: 9px;
    }

    .efficiency-track {
      height: 17px;
      background: rgba(120,170,255,0.10);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 20px;
    }

    .efficiency-fill {
      height: 100%;
      border-radius: 10px;
      transition: width 0.3s ease;
    }

    .efficiency-fill.dense {
      background: #43e7ff;
    }

    .efficiency-fill.sparse {
      background: #ad7bff;
    }

    .efficiency-insight {
      padding: 18px;
      border: 1px solid rgba(67,231,255,0.16);
      border-radius: 15px;
      background: rgba(67,231,255,0.07);
      line-height: 1.7;
      color: #dce9ff;
      font-size: 14px;
    }

    .efficiency-footnote {
      margin-top: 16px;
      color: #97a8c3;
      font-size: 12px;
      line-height: 1.6;
    }

    @media (max-width: 1100px) {
      .efficiency-lab {
        grid-column: 1 / -1;
      }
    }

    @media (max-width: 600px) {
      .efficiency-controls,
      .efficiency-metrics {
        grid-template-columns: 1fr;
      }
    }
  `;

  document.head.appendChild(styles);

  const section = document.createElement("section");
  section.className = "card efficiency-lab";

  section.innerHTML = `
    <h2>Inference Efficiency Lab</h2>

    <p class="efficiency-subtitle">
      Fewer operations do not necessarily mean faster execution.
      Compare real AQUA inference benchmarks across trained
      checkpoints and batch sizes.
    </p>

    <div class="efficiency-controls">
      <div>
        <label for="efficiencyCheckpoint">Recorded Checkpoint</label>
        <select id="efficiencyCheckpoint"></select>
      </div>

      <div>
        <label for="efficiencyBatch">Batch Size</label>
        <select id="efficiencyBatch"></select>
      </div>
    </div>

    <div class="efficiency-metrics">
      <div class="efficiency-metric">
        <span>Weight Multiplication Reduction</span>
        <strong id="effArithmetic">--</strong>
      </div>

      <div class="efficiency-metric">
        <span>Sparse Array Storage</span>
        <strong id="effStorage">--</strong>
      </div>

      <div class="efficiency-metric">
        <span>Speed Relative to Dense</span>
        <strong id="effSpeedup">--</strong>
      </div>
    </div>

    <div class="efficiency-chart">
      <h3>Measured Inference Latency</h3>

      <div class="efficiency-bar-label">
        <span>Optimized Dense NumPy</span>
        <strong id="effDenseTime">--</strong>
      </div>
      <div class="efficiency-track">
        <div class="efficiency-fill dense" id="effDenseBar"></div>
      </div>

      <div class="efficiency-bar-label">
        <span>Sparse-aware Python</span>
        <strong id="effSparseTime">--</strong>
      </div>
      <div class="efficiency-track">
        <div class="efficiency-fill sparse" id="effSparseBar"></div>
      </div>
    </div>

    <div class="efficiency-insight" id="effInsight"></div>

    <p class="efficiency-footnote">
      Results are from recorded inference benchmarks.
      Sparse array storage includes stored indices and numerical arrays.
      Latency reflects the measured Python/NumPy implementation,
      not a hardware-optimized sparse kernel.
    </p>
  `;

  // Position after Method Comparison.
  const comparisonCard = document.querySelector(".comparison-card");

  if (comparisonCard) {
    comparisonCard.insertAdjacentElement("afterend", section);
  } else {
    dashboard.appendChild(section);
  }

  const checkpointSelect =
    document.getElementById("efficiencyCheckpoint");

  const batchSelect =
    document.getElementById("efficiencyBatch");

  const checkpoints = [...new Set(
    rows.map(row => row.checkpoint)
  )];

  const batches = [...new Set(
    rows.map(row => Number(row.batch_size))
  )].sort((a, b) => a - b);

  for (const checkpoint of checkpoints) {
    checkpointSelect.add(new Option(
      checkpoint.replaceAll("_", " "),
      checkpoint
    ));
  }

  for (const batch of batches) {
    batchSelect.add(new Option(
      String(batch),
      String(batch)
    ));
  }

  // Prefer a pruned checkpoint if present.
  const preferred = checkpoints.find(
    name => name.includes("taylor_90")
  ) || checkpoints.find(
    name => !name.includes("_0_")
  );

  if (preferred) checkpointSelect.value = preferred;

  batchSelect.value = batches.includes(32) ? "32" : String(batches[0]);

  function renderEfficiency() {
    const checkpoint = checkpointSelect.value;
    const batch = Number(batchSelect.value);

    const row = rows.find(
      item =>
        item.checkpoint === checkpoint &&
        Number(item.batch_size) === batch
    );

    if (!row) return;

    const dense = Number(row.optimized_dense_latency_ms);
    const sparse = Number(row.sparse_latency_ms);
    const speedup = Number(row.speedup_vs_optimized_dense);

    const reduction = Number(row.arithmetic_reduction);

    const denseBytes = Number(row.dense_parameter_bytes);
    const sparseBytes = Number(row.sparse_array_bytes);

    document.getElementById("effArithmetic").textContent =
      `${(reduction * 100).toFixed(2)}%`;

    document.getElementById("effStorage").textContent =
      `${sparseBytes.toLocaleString()} bytes`;

    document.getElementById("effSpeedup").textContent =
      `${speedup.toFixed(3)}×`;

    document.getElementById("effDenseTime").textContent =
      `${dense.toFixed(4)} ms`;

    document.getElementById("effSparseTime").textContent =
      `${sparse.toFixed(4)} ms`;

    const maximum = Math.max(dense, sparse);

    document.getElementById("effDenseBar").style.width =
      `${(dense / maximum) * 100}%`;

    document.getElementById("effSparseBar").style.width =
      `${(sparse / maximum) * 100}%`;

    const insight = document.getElementById("effInsight");

    const speedDescription = speedup < 1
      ? `Sparse-aware inference was ${(sparse / dense).toFixed(2)}× slower than optimized dense NumPy.`
      : `Sparse-aware inference was ${speedup.toFixed(2)}× faster than optimized dense NumPy.`;

    const storageDifference = sparseBytes - denseBytes;

    const storageDescription = storageDifference > 0
      ? `The sparse representation used ${storageDifference.toLocaleString()} more bytes than the dense parameter arrays.`
      : `The sparse representation saved ${Math.abs(storageDifference).toLocaleString()} bytes relative to the dense parameter arrays.`;

    insight.textContent =
      `This checkpoint eliminated ${(reduction * 100).toFixed(2)}% ` +
      `of weight multiplications for batch size ${batch}. ` +
      speedDescription + " " + storageDescription;
  }

  checkpointSelect.addEventListener("change", renderEfficiency);
  batchSelect.addEventListener("change", renderEfficiency);

  renderEfficiency();
})().catch(error => {
  console.error("AQUA Efficiency Lab error:", error);
});
