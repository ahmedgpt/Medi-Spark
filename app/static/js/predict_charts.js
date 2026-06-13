/**
 * Predict page: Severity gauge chart + Print/Download buttons
 * Hooks into the predict output after renderResult() runs.
 */

// Watch for result rendering and enhance it
const predictOutput = document.getElementById('predict-output');
if (predictOutput) {
  const observer = new MutationObserver(() => {
    const result = predictOutput.querySelector('.result');
    if (!result) return;

    // Don't double-inject
    if (result.querySelector('.gauge-section')) return;

    // ── Extract severity from the rendered badge ───────────────────────────
    const badge = result.querySelector('.risk-badge');
    let severity = 0;
    if (badge) {
      const match = badge.textContent.match(/Severity\s+(\d+)/);
      if (match) severity = parseInt(match[1], 10);
    }

    // ── Inject severity gauge ──────────────────────────────────────────────
    const gaugeSection = document.createElement('div');
    gaugeSection.className = 'gauge-section';
    gaugeSection.innerHTML = `
      <div class="inline-gauge">
        <canvas id="predict-gauge" width="140" height="140"></canvas>
        <div class="gauge-center small">${severity}/100</div>
      </div>
    `;
    result.insertBefore(gaugeSection, result.firstChild.nextSibling);

    // Draw gauge
    const ctx = document.getElementById('predict-gauge');
    if (ctx && typeof Chart !== 'undefined') {
      const color = severity >= 70 ? '#c0392b' : severity >= 40 ? '#d39e00' : '#1f8a4c';
      new Chart(ctx, {
        type: 'doughnut',
        data: {
          datasets: [{
            data: [severity, 100 - severity],
            backgroundColor: [color, '#e9ecef'],
            borderWidth: 0,
          }],
        },
        options: {
          cutout: '72%',
          responsive: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          animation: { animateRotate: true, duration: 1000 },
        },
      });
    }

    // ── Inject Print / Download buttons ────────────────────────────────────
    const actions = document.createElement('div');
    actions.className = 'result-actions';
    actions.innerHTML = `
      <button class="btn outline small" id="btn-print" title="Print this result">🖨️ Print</button>
      <button class="btn outline small" id="btn-download" title="Download as text file">📥 Download</button>
    `;
    result.appendChild(actions);

    document.getElementById('btn-print').addEventListener('click', () => {
      window.print();
    });

    document.getElementById('btn-download').addEventListener('click', () => {
      const text = result.innerText;
      const blob = new Blob(
        ['MediSpark — Symptom Analysis Report\n' + '='.repeat(40) + '\n\n' + text],
        { type: 'text/plain' }
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `medispark_report_${new Date().toISOString().slice(0, 10)}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    });
  });

  observer.observe(predictOutput, { childList: true, subtree: true });
}
