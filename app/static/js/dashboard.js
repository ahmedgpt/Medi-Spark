/**
 * MediSpark Dashboard — Chart.js charts + stats
 * Fetches /api/dashboard/stats and renders severity gauge, risk pie, disease bar, severity timeline.
 */

const COLORS = {
  primary: '#1f7a8c',
  primaryDark: '#155060',
  danger: '#c0392b',
  warning: '#d39e00',
  success: '#1f8a4c',
  muted: '#6b7280',
  bg: '#f4f7fb',
};

// ── Fetch stats ──────────────────────────────────────────────────────────────

async function loadDashboard() {
  let stats;
  try {
    const res = await fetch('/api/dashboard/stats');
    stats = await res.json();
  } catch {
    stats = {
      total_consultations: 0,
      risk_distribution: { HIGH: 0, MEDIUM: 0, LOW: 0 },
      recent_severity: 0,
      recent_disease: '—',
      recent_risk: 'UNKNOWN',
      disease_trend: [],
      severity_history: [],
    };
  }

  // ── Update stat cards ────────────────────────────────────────────────────
  document.getElementById('stat-total').textContent = stats.total_consultations;
  document.getElementById('stat-disease').textContent = stats.recent_disease;
  document.getElementById('stat-severity').textContent = stats.recent_severity + '/100';
  document.getElementById('stat-risk').textContent = stats.recent_risk;

  const riskIcon = document.getElementById('stat-risk-icon');
  const riskMap = { HIGH: '🔴', MEDIUM: '🟡', LOW: '🟢', UNKNOWN: '⚪' };
  riskIcon.textContent = riskMap[stats.recent_risk] || '⚪';

  // ── Severity Gauge (Doughnut) ────────────────────────────────────────────
  const sev = stats.recent_severity || 0;
  const gaugeLabel = document.getElementById('gauge-label');
  gaugeLabel.textContent = sev + '/100';
  gaugeLabel.style.color = sev >= 70 ? COLORS.danger : sev >= 40 ? COLORS.warning : COLORS.success;

  const gaugeCtx = document.getElementById('severity-gauge');
  if (gaugeCtx) {
    new Chart(gaugeCtx, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [sev, 100 - sev],
          backgroundColor: [
            sev >= 70 ? COLORS.danger : sev >= 40 ? COLORS.warning : COLORS.success,
            '#e9ecef',
          ],
          borderWidth: 0,
        }],
      },
      options: {
        cutout: '75%',
        responsive: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { animateRotate: true, duration: 1200 },
      },
    });
  }

  // ── Risk Distribution (Doughnut) ─────────────────────────────────────────
  const riskCtx = document.getElementById('risk-chart');
  if (riskCtx) {
    const rd = stats.risk_distribution;
    new Chart(riskCtx, {
      type: 'doughnut',
      data: {
        labels: ['High', 'Medium', 'Low'],
        datasets: [{
          data: [rd.HIGH || 0, rd.MEDIUM || 0, rd.LOW || 0],
          backgroundColor: [COLORS.danger, COLORS.warning, COLORS.success],
          borderWidth: 2,
          borderColor: '#fff',
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { padding: 16, font: { family: 'Inter', size: 12 } } },
        },
        animation: { duration: 1000 },
      },
    });
  }

  // ── Disease Trend (Bar) ──────────────────────────────────────────────────
  const diseaseCtx = document.getElementById('disease-chart');
  if (diseaseCtx && stats.disease_trend.length) {
    new Chart(diseaseCtx, {
      type: 'bar',
      data: {
        labels: stats.disease_trend.map(d => d.disease),
        datasets: [{
          label: 'Cases',
          data: stats.disease_trend.map(d => d.count),
          backgroundColor: [
            COLORS.primary, COLORS.primaryDark, COLORS.success, COLORS.warning, COLORS.danger,
          ],
          borderRadius: 6,
          barPercentage: 0.6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: { grid: { display: false }, ticks: { font: { family: 'Inter' } } },
          y: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 11 } } },
        },
        animation: { duration: 800 },
      },
    });
  } else if (diseaseCtx) {
    diseaseCtx.parentElement.innerHTML += '<p class="no-data">No prediction data yet. Run a symptom check to see trends.</p>';
  }

  // ── Severity Over Time (Line) ────────────────────────────────────────────
  const sevHistCtx = document.getElementById('severity-history-chart');
  if (sevHistCtx && stats.severity_history.length) {
    new Chart(sevHistCtx, {
      type: 'line',
      data: {
        labels: stats.severity_history.map(d => d.date),
        datasets: [{
          label: 'Severity',
          data: stats.severity_history.map(d => d.severity),
          borderColor: COLORS.primary,
          backgroundColor: COLORS.primary + '20',
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointBackgroundColor: COLORS.primary,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          y: { min: 0, max: 100, ticks: { font: { family: 'Inter' } } },
          x: { ticks: { font: { family: 'Inter', size: 10 } } },
        },
        animation: { duration: 1000 },
      },
    });
  } else if (sevHistCtx) {
    sevHistCtx.parentElement.innerHTML += '<p class="no-data">No severity history yet.</p>';
  }
}

document.addEventListener('DOMContentLoaded', loadDashboard);
