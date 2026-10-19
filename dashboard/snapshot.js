
/*
 * AQUA Observatory — Connection Survival
 *
 * Uses the actual metrics displayed by app.js.
 * Does not modify training code or experiment data.
 */

(function initializeSnapshot() {
  const metricsCard = document.querySelector(".metrics-card");

  const activeElement = document.getElementById("activeWeights");
  const removedElement = document.getElementById("removedWeights");

  if (!metricsCard || !activeElement || !removedElement) {
    console.error("AQUA Snapshot: Required elements not found.");
    return;
  }

  // --------------------------------------------------
  // 1. Add styling
  // --------------------------------------------------

  const style = document.createElement("style");

  style.textContent = `
    .aqua-survival {
      margin-top: 26px;
      padding: 20px;
      border: 1px solid rgba(120, 170, 255, 0.14);
      border-radius: 17px;
      background: rgba(7, 14, 26, 0.55);
    }

    .aqua-survival-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }

    .aqua-survival-header strong {
      color: #e9f1ff;
      font-size: 14px;
    }

    .aqua-survival-header span {
      color: #97a8c3;
      font-size: 12px;
    }

    .aqua-survival-track {
      display: flex;
      overflow: hidden;
      width: 100%;
      height: 15px;
      border-radius: 999px;
      background: #52637b;
    }

    .aqua-survival-active {
      background: #43e7ff;
      height: 100%;
      transition: width 350ms ease;
      box-shadow: 0 0 12px rgba(67, 231, 255, 0.3);
    }

    .aqua-survival-removed {
      background: #52637b;
      height: 100%;
      transition: width 350ms ease;
    }

    .aqua-survival-counts {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
      color: #97a8c3;
      font-size: 12px;
    }

    .aqua-survival-note {
      margin-top: 17px;
      color: #c6d5e9;
      line-height: 1.6;
      font-size: 13px;
    }

    .aqua-network-legend {
      margin-top: 24px;
      padding-top: 20px;
      border-top: 1px solid rgba(120, 170, 255, 0.13);
    }

    .aqua-network-legend-title {
      margin-bottom: 13px;
      color: #e9f1ff;
      font-size: 13px;
      font-weight: 700;
    }

    .aqua-legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 11px;
      color: #97a8c3;
      font-size: 12px;
    }

    .aqua-legend-line {
      width: 30px;
      height: 3px;
      border-radius: 4px;
      flex-shrink: 0;
    }

    .aqua-legend-line.active {
      background: #43e7ff;
      box-shadow: 0 0 9px rgba(67, 231, 255, 0.4);
    }

    .aqua-legend-line.pruned {
      background: #52637b;
      opacity: 0.35;
    }

    .aqua-legend-footnote {
      color: #97a8c3;
      font-size: 11px;
      line-height: 1.5;
      margin-top: 14px;
    }
  `;

  document.head.appendChild(style);

  // --------------------------------------------------
  // 2. Build visualization
  // --------------------------------------------------

  const panel = document.createElement("div");
  panel.className = "aqua-survival";

  panel.innerHTML = `
    <div class="aqua-survival-header">
      <strong>Connection Survival</strong>
      <span>128 total weights</span>
    </div>

    <div
      class="aqua-survival-track"
      role="img"
      aria-label="Proportion of active and removed connections"
      id="aquaSurvivalTrack"
    >
      <div
        class="aqua-survival-active"
        id="aquaSurvivalActive"
      ></div>

      <div
        class="aqua-survival-removed"
        id="aquaSurvivalRemoved"
      ></div>
    </div>

    <div class="aqua-survival-counts">
      <span id="aquaSurvivalActiveCount">-- active</span>
      <span id="aquaSurvivalRemovedCount">-- removed</span>
    </div>

    <div class="aqua-survival-note" id="aquaSurvivalNote">
      Waiting for recorded experiment data.
    </div>

    <div class="aqua-network-legend">
      <div class="aqua-network-legend-title">
        Reading the Network
      </div>

      <div class="aqua-legend-item">
        <span class="aqua-legend-line active"></span>
        <span>Active weight connection</span>
      </div>

      <div class="aqua-legend-item">
        <span class="aqua-legend-line pruned"></span>
        <span>Pruned weight connection</span>
      </div>

      <p class="aqua-legend-footnote">
        This visualization represents the final trained
        checkpoint. Hidden neurons are shown even when
        their individual connections have been pruned.
      </p>
    </div>
  `;

  metricsCard.appendChild(panel);

  // --------------------------------------------------
  // 3. Update using real recorded counts
  // --------------------------------------------------

  function updateSurvival() {
    const active = Number(activeElement.textContent.trim());
    const removed = Number(removedElement.textContent.trim());

    if (!Number.isFinite(active) || !Number.isFinite(removed)) {
      return;
    }

    const total = active + removed;

    if (total !== 128) {
      return;
    }

    const activePercentage = (active / total) * 100;
    const removedPercentage = (removed / total) * 100;

    document.getElementById("aquaSurvivalActive").style.width =
      `${activePercentage}%`;

    document.getElementById("aquaSurvivalRemoved").style.width =
      `${removedPercentage}%`;

    document.getElementById("aquaSurvivalActiveCount").textContent =
      `${active} active`;

    document.getElementById("aquaSurvivalRemovedCount").textContent =
      `${removed} removed`;

    document.getElementById("aquaSurvivalNote").textContent =
      `${activePercentage.toFixed(2)}% of the network's ` +
      `weight connections remain active in this checkpoint.`;

    document.getElementById("aquaSurvivalTrack").setAttribute(
      "aria-label",
      `${active} active and ${removed} removed weight connections`
    );
  }

  // The main app updates metric text whenever the
  // experiment selector changes. Observe those changes.
  const observer = new MutationObserver(updateSurvival);

  observer.observe(activeElement, {
    childList: true,
    characterData: true,
    subtree: true
  });

  observer.observe(removedElement, {
    childList: true,
    characterData: true,
    subtree: true
  });

  updateSurvival();
})();
