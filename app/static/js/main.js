// Week 1 — submits the symptom form to /api/predict and prints the JSON ack.
const form = document.getElementById('predict-form');
const out = document.getElementById('predict-output');
if (form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    out.textContent = 'Submitting…';
    const data = Object.fromEntries(new FormData(form).entries());
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const json = await res.json();
      out.textContent = JSON.stringify(json, null, 2);
    } catch (err) {
      out.textContent = 'Error: ' + err.message;
    }
  });
}
