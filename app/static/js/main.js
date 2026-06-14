// Submits the symptom form to /api/predict and renders a readable result.

const form = document.getElementById('predict-form');
const out  = document.getElementById('predict-output');

// Lightweight Roman-Urdu → English medical dictionary used as a client-side
// fallback when the user picks "Roman Urdu" (full translator lands in Week 3).
const URDU_MEDICAL_DICT = {
  'bukhaar': 'fever', 'bukhar': 'fever', 'taiz_bukhaar': 'high_fever',
  'sar_dard': 'headache', 'sardard': 'headache',
  'khansi': 'cough', 'kamzori': 'fatigue',
  'dast': 'diarrhoea', 'ulti': 'vomiting', 'matli': 'nausea',
  'pait_dard': 'stomach_pain', 'seene_mein_dard': 'chest_pain',
  'saans_phulna': 'difficulty_breathing', 'chakkar': 'dizziness',
  'jor_dard': 'joint_pain', 'jism_dard': 'muscle_pain',
};

function parseSymptoms(text, language) {
  const tokens = (text || '')
    .split(/[,\n;]+/)
    .map(s => s.trim().toLowerCase())
    .filter(Boolean);
  if (language !== 'ur') return tokens;
  // For Urdu, also try underscore form for dictionary lookup
  return tokens.map(t => {
    const key = t.replace(/\s+/g, '_');
    return URDU_MEDICAL_DICT[key] || URDU_MEDICAL_DICT[t] || t;
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function renderResult(json) {
  if (json.error) {
    out.innerHTML = `<div class="error">⚠ ${escapeHtml(json.error)}</div>`;
    return;
  }

  const r = json.risk || {};
  const badgeClass = (r.risk_level || '').toLowerCase();
  const preds = (json.predictions || [])
    .map(p => `<li><strong>${escapeHtml(p.disease)}</strong> — ${(p.confidence * 100).toFixed(1)}%</li>`)
    .join('');

  const tests = (r.recommended_tests || []).length
    ? `<ul>${r.recommended_tests.map(t => `<li>${escapeHtml(t)}</li>`).join('')}</ul>`
    : '<em>None</em>';

  const meds = json.medicines;
  let medsHtml = '<em>None</em>';
  if (Array.isArray(meds) && meds.length) {
    medsHtml = `<ul>${meds.map(m => `<li>${escapeHtml(typeof m === 'string' ? m : JSON.stringify(m))}</li>`).join('')}</ul>`;
  } else if (meds && typeof meds === 'object') {
    medsHtml = `<pre>${escapeHtml(JSON.stringify(meds, null, 2))}</pre>`;
  }

  const rag = (json.rag_knowledge || [])
    .map(k => `<li><strong>${escapeHtml(k.disease)}</strong>: ${escapeHtml(k.content)}…</li>`)
    .join('');

  const summaryHtml = json.llm_summary
    ? `<div class="alert alert-info mt-3"><strong>AI Advisory:</strong> ${escapeHtml(json.llm_summary)}</div>`
    : '';

  out.innerHTML = `
    <div class="result">
      <div class="risk-badge ${badgeClass}">
        ${escapeHtml(r.risk_level || 'UNKNOWN')} · Severity ${r.severity_score ?? '–'}/100
      </div>
      ${summaryHtml}
      <h3>Top predictions</h3>
      <ol>${preds || '<li>No predictions returned.</li>'}</ol>
      <h3>Advice</h3>
      <p>${escapeHtml(r.advice || '')}</p>
      <h3>Recommended tests</h3>
      ${tests}
      <h3>Suggested medicines</h3>
      ${medsHtml}
      ${rag ? `<h3>Medical knowledge</h3><ul>${rag}</ul>` : ''}
    </div>`;
}

if (form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    out.textContent = 'Analysing…';

    const fd       = new FormData(form);
    const language = fd.get('language') || 'en';
    const symptoms = parseSymptoms(fd.get('symptoms_text'), language);

    if (symptoms.length === 0) {
      out.innerHTML = '<div class="error">Please enter at least one symptom.</div>';
      return;
    }

    const payload = {
      symptoms,
      duration_days: parseInt(fd.get('duration_days'), 10) || 1,
      age:           parseInt(fd.get('age'), 10) || 30,
      language,
    };

    try {
      const res  = await fetch('/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });
      const json = await res.json();
      renderResult(json);
    } catch (err) {
      out.innerHTML = `<div class="error">Network error: ${escapeHtml(err.message)}</div>`;
    }
  });
}
