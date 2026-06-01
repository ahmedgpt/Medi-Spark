/**
 * MediSpark Chat UI — Week 3
 * Handles message sending, bot replies, language badges, reset, and auto-scroll.
 */

const chatForm     = document.getElementById('chat-form');
const chatInput    = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const resetBtn     = document.getElementById('chat-reset-btn');
const langBadge    = document.getElementById('chat-lang-badge');

// ── Helpers ──────────────────────────────────────────────────────────────────

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
  );
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatBotReply(text) {
  // Convert **bold** to <strong>
  let html = escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
  return html;
}

function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// ── Message rendering ────────────────────────────────────────────────────────

function addMessage(role, content, extra = {}) {
  const wrapper = document.createElement('div');
  wrapper.classList.add('message', role);

  const avatar = document.createElement('div');
  avatar.classList.add('message-avatar');
  avatar.textContent = role === 'user' ? '👤' : '🩺';

  const bubble = document.createElement('div');
  bubble.classList.add('message-bubble');

  if (role === 'user') {
    bubble.innerHTML = `<p>${escapeHtml(content)}</p>`;
  } else {
    bubble.innerHTML = `<p>${formatBotReply(content)}</p>`;
  }

  // Show translation if input was non-English
  if (extra.translated && role === 'bot') {
    const transEl = document.createElement('p');
    transEl.classList.add('translation-note');
    transEl.innerHTML = `🌐 <em>Translated from ${escapeHtml(extra.detected_lang)}: "${escapeHtml(extra.translated)}"</em>`;
    bubble.prepend(transEl);
  }

  // Show RAG sources if available
  if (extra.sources && extra.sources.length > 0) {
    const srcEl = document.createElement('details');
    srcEl.classList.add('sources-details');
    srcEl.innerHTML = `<summary>📚 Medical sources (${extra.sources.length})</summary>` +
      extra.sources.map(s => `<p class="source-snippet">${escapeHtml(s.substring(0, 250))}…</p>`).join('');
    bubble.appendChild(srcEl);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function addTypingIndicator() {
  const wrapper = document.createElement('div');
  wrapper.classList.add('message', 'bot', 'typing-indicator');
  wrapper.id = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.classList.add('message-avatar');
  avatar.textContent = '🩺';

  const bubble = document.createElement('div');
  bubble.classList.add('message-bubble');
  bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

// ── Language badge ───────────────────────────────────────────────────────────

function showLangBadge(lang) {
  if (!lang || lang === 'english') {
    langBadge.style.display = 'none';
    return;
  }
  const labels = {
    'roman_urdu': '🇵🇰 Roman Urdu detected',
    'urdu': '🇵🇰 Urdu detected',
  };
  langBadge.textContent = labels[lang] || lang;
  langBadge.style.display = 'inline-block';
  setTimeout(() => { langBadge.style.display = 'none'; }, 5000);
}

// ── Send message ─────────────────────────────────────────────────────────────

async function sendMessage(message) {
  if (!message.trim()) return;

  // Show user message
  addMessage('user', message);
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // Show typing indicator
  addTypingIndicator();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    removeTypingIndicator();

    if (data.error) {
      addMessage('bot', `⚠️ Error: ${data.error}`);
      return;
    }

    showLangBadge(data.detected_lang);
    addMessage('bot', data.reply, {
      translated: data.translated,
      detected_lang: data.detected_lang,
      sources: data.sources,
    });
  } catch (err) {
    removeTypingIndicator();
    addMessage('bot', `⚠️ Network error: ${err.message}. Please try again.`);
  }
}

// ── Reset conversation ───────────────────────────────────────────────────────

async function resetConversation() {
  try {
    await fetch('/api/chat/reset', { method: 'DELETE' });
  } catch (e) { /* ignore */ }

  // Clear all messages except the welcome message
  const messages = chatMessages.querySelectorAll('.message');
  messages.forEach((msg, i) => {
    if (i > 0) msg.remove();   // keep first (welcome)
  });
}

// ── Load chat history on page load ───────────────────────────────────────────

async function loadHistory() {
  try {
    const res = await fetch('/api/chat/history');
    const data = await res.json();
    if (data.history && data.history.length > 0) {
      data.history.forEach(entry => {
        addMessage(entry.role === 'human' ? 'user' : 'bot', entry.content);
      });
    }
  } catch (e) { /* ignore — history will be empty */ }
}

// ── Event listeners ──────────────────────────────────────────────────────────

if (chatForm) {
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage(chatInput.value);
  });
}

if (chatInput) {
  chatInput.addEventListener('input', () => autoResize(chatInput));

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(chatInput.value);
    }
  });
}

if (resetBtn) {
  resetBtn.addEventListener('click', () => {
    if (confirm('Clear entire conversation? This cannot be undone.')) {
      resetConversation();
    }
  });
}

// Load history when page loads
document.addEventListener('DOMContentLoaded', loadHistory);
