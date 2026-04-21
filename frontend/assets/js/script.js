/**
 * script.js — PlagiarismCheck frontend
 *
 * Responsibilities:
 *  - Tab switching (text / file mode)
 *  - Form validation
 *  - Fetch calls to the FastAPI backend
 *  - Rendering the score card, breakdown, matched pairs, and highlights
 *  - Drag-and-drop file upload
 *  - Health check indicator
 *  - Error / loading states
 */

'use strict';

/* ══════════════════════════════════════════════
   CONFIG
══════════════════════════════════════════════ */

const API_BASE = window.location.origin; // Same origin — served by FastAPI

const ENDPOINTS = {
  health:         `${API_BASE}/api/health`,
  checkPlagiarism:`${API_BASE}/api/check-plagiarism`,
  checkFiles:     `${API_BASE}/api/check-files`,
};

/* Loading message carousel — cycles while waiting */
const LOADING_MESSAGES = [
  'Tokenising and lemmatising…',
  'Building shingle fingerprints…',
  'Computing TF-IDF vectors…',
  'Running cosine similarity…',
  'Aligning sentence pairs…',
  'Finalising report…',
];

/* ══════════════════════════════════════════════
   DOM REFERENCES
══════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

const el = {
  // Tabs
  tabBtns:    document.querySelectorAll('.tab-btn'),
  navLinks:   document.querySelectorAll('.nav-link[data-mode]'),
  panels:     { text: $('panel-text'), file: $('panel-file') },

  // Text mode
  textA:      $('textA'),
  textB:      $('textB'),
  countA:     $('countA'),
  countB:     $('countB'),
  clearA:     $('clearA'),
  clearB:     $('clearB'),
  btnCheckText: $('btnCheckText'),
  btnLoadSample: $('btnLoadSample'),

  // File mode
  dropA:      $('dropA'),
  dropB:      $('dropB'),
  fileInputA: $('fileInputA'),
  fileInputB: $('fileInputB'),
  filenameA:  $('filenameA'),
  filenameB:  $('filenameB'),
  btnCheckFiles: $('btnCheckFiles'),

  // Loader
  loader:     $('loader'),
  loaderText: $('loaderText'),

  // Error
  errorBanner:$('errorBanner'),
  errorMsg:   $('errorMsg'),
  errorClose: $('errorClose'),

  // Results
  resultsSection: $('resultsSection'),
  scoreMain:  $('scoreMain'),
  scoreBar:   $('scoreBar'),
  scoreVerdict: $('scoreVerdict'),
  scoreCard:  document.querySelector('.score-card'),

  barJaccard: $('barJaccard'),
  barTfidf:   $('barTfidf'),
  barSemantic:$('barSemantic'),
  valJaccard: $('valJaccard'),
  valTfidf:   $('valTfidf'),
  valSemantic:$('valSemantic'),
  rowSemantic:$('rowSemantic'),
  metaSemStatus: $('metaSemStatus'),
  metaTime:   $('metaTime'),

  matchesList:$('matchesList'),
  matchCount: $('matchCount'),
  noMatches:  $('noMatches'),
  toggleSemOnly: $('toggleSemOnly'),

  paneA:      $('paneA'),
  paneB:      $('paneB'),

  btnReset:   $('btnReset'),
  healthDot:  $('healthDot'),
};

/* ══════════════════════════════════════════════
   STATE
══════════════════════════════════════════════ */

let currentMatchedPairs = [];  // stored so we can re-filter
let loadingTimer = null;

/* ══════════════════════════════════════════════
   HEALTH CHECK
══════════════════════════════════════════════ */

async function checkHealth() {
  try {
    const res = await fetch(ENDPOINTS.health, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      el.healthDot.classList.add('online');
      el.healthDot.title = 'API is online';
      el.healthDot.querySelector('.health-label').textContent = 'API';
    } else {
      throw new Error('non-200');
    }
  } catch {
    el.healthDot.classList.add('offline');
    el.healthDot.title = 'API is offline — start the FastAPI server';
  }
}

/* ══════════════════════════════════════════════
   TAB SWITCHING
══════════════════════════════════════════════ */

function switchTab(mode) {
  el.tabBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === mode));
  el.navLinks.forEach(link => link.classList.toggle('active', link.dataset.mode === mode));
  Object.keys(el.panels).forEach(k => {
    el.panels[k].classList.toggle('active', k === mode);
  });
  hideResults();
  hideError();
}

el.tabBtns.forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));
el.navLinks.forEach(link => link.addEventListener('click', e => {
  e.preventDefault();
  switchTab(link.dataset.mode);
}));

/* ══════════════════════════════════════════════
   CHAR COUNT
══════════════════════════════════════════════ */

function updateCount(textarea, counter) {
  const n = textarea.value.length;
  counter.textContent = `${n.toLocaleString()} character${n !== 1 ? 's' : ''}`;
}

el.textA.addEventListener('input', () => updateCount(el.textA, el.countA));
el.textB.addEventListener('input', () => updateCount(el.textB, el.countB));

el.clearA.addEventListener('click', () => { el.textA.value = ''; updateCount(el.textA, el.countA); });
el.clearB.addEventListener('click', () => { el.textB.value = ''; updateCount(el.textB, el.countB); });

/* ══════════════════════════════════════════════
   SAMPLE DATA
══════════════════════════════════════════════ */

const SAMPLE = {
  a: `Artificial intelligence is transforming every sector of the modern economy. Machine learning algorithms can now process vast amounts of data to identify patterns that would be invisible to human analysts. Deep learning, a subset of machine learning, has proven particularly effective for image recognition, natural language processing, and autonomous decision-making systems. Companies that invest in AI research early are likely to gain significant competitive advantages.`,

  b: `The modern economy is being reshaped by artificial intelligence in profound ways. Advanced learning algorithms are capable of analysing enormous datasets, uncovering patterns beyond human perception. Neural networks and deep learning approaches have shown remarkable results in recognising images, understanding language, and enabling systems to make decisions without human intervention. Organisations that adopt AI technology ahead of their competitors stand to benefit greatly.`,
};

el.btnLoadSample.addEventListener('click', () => {
  el.textA.value = SAMPLE.a;
  el.textB.value = SAMPLE.b;
  updateCount(el.textA, el.countA);
  updateCount(el.textB, el.countB);
  hideResults();
  hideError();
});

/* ══════════════════════════════════════════════
   FILE UPLOAD — drag-and-drop + click
══════════════════════════════════════════════ */

function setupFileZone(dropZone, fileInput, filenameEl) {
  // Click handled by <label for="...">
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file && file.size > 5 * 1024 * 1024) {
      showError("File too large (max 5MB)");
      return;
}
    filenameEl.textContent = file ? `✓ ${file.name}` : '';
  });

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (!file) return;

    // Transfer file to the hidden input
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    filenameEl.textContent = `✓ ${file.name}`;
  });
}

setupFileZone(el.dropA, el.fileInputA, el.filenameA);
setupFileZone(el.dropB, el.fileInputB, el.filenameB);

/* ══════════════════════════════════════════════
   LOADING STATE
══════════════════════════════════════════════ */

function showLoader() {
  el.loader.hidden = false;
  el.resultsSection.hidden = true;
  hideError();

  let msgIdx = 0;
  el.loaderText.textContent = LOADING_MESSAGES[0];
  loadingTimer = setInterval(() => {
    msgIdx = (msgIdx + 1) % LOADING_MESSAGES.length;
    el.loaderText.textContent = LOADING_MESSAGES[msgIdx];
  }, 1800);
}

function hideLoader() {
  el.loader.hidden = true;
  clearInterval(loadingTimer);
  loadingTimer = null;
}

/* ══════════════════════════════════════════════
   ERROR HANDLING
══════════════════════════════════════════════ */

function showError(msg) {
  el.errorMsg.textContent = msg;
  el.errorBanner.hidden = false;
  el.resultsSection.hidden = true;
}

function hideError() {
  el.errorBanner.hidden = true;
}

el.errorClose.addEventListener('click', hideError);

/* ══════════════════════════════════════════════
   API CALLS
══════════════════════════════════════════════ */

/**
 * POST /api/check-plagiarism with JSON body.
 */
async function apiCheckText(textA, textB) {
  const res = await fetch(ENDPOINTS.checkPlagiarism, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text_a: textA, text_b: textB }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

/**
 * POST /api/check-files with multipart form data.
 */
async function apiCheckFiles(fileA, fileB) {
  const form = new FormData();
  form.append('file_a', fileA);
  form.append('file_b', fileB);

  const res = await fetch(ENDPOINTS.checkFiles, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

/* ══════════════════════════════════════════════
   BUTTON HANDLERS
══════════════════════════════════════════════ */

el.btnCheckText.addEventListener('click', async () => {
  const a = el.textA.value.trim();
  const b = el.textB.value.trim();

  if (a.length < 20) { showError('Document A must be at least 20 characters.'); return; }
  if (b.length < 20) { showError('Document B must be at least 20 characters.'); return; }

  showLoader();
  el.btnCheckText.disabled = true;

  try {
    const data = await apiCheckText(a, b);
    renderResults(data, a, b);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoader();
    el.btnCheckText.disabled = false;
  }
});

el.btnCheckFiles.addEventListener('click', async () => {
  const fileA = el.fileInputA.files[0];
  const fileB = el.fileInputB.files[0];

  if (!fileA) { showError('Please select Document A.'); return; }
  if (!fileB) { showError('Please select Document B.'); return; }

  showLoader();
  el.btnCheckFiles.disabled = true;

  try {
    const data = await apiCheckFiles(fileA, fileB);
    renderResults(data, null, null);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoader();
    el.btnCheckFiles.disabled = false;
  }
});

/* ══════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════ */

/**
 * Main render function — populates every section of the results panel.
 *
 * @param {Object}      data   API response object.
 * @param {string|null} textA  Original text A (null for file mode).
 * @param {string|null} textB  Original text B (null for file mode).
 */
function renderResults(data, textA, textB) {
  currentMatchedPairs = data.matched_pairs || [];

  renderScoreCard(data);
  renderBreakdown(data);
  renderMatches(currentMatchedPairs);

  if (textA && textB) {
    renderHighlights(textA, textB, currentMatchedPairs);
    document.querySelector('.highlight-section').hidden = false;
  } else {
    document.querySelector('.highlight-section').hidden = true;
  }

  el.resultsSection.hidden = false;
  el.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ─── Score card ─────────────────────────────────────────── */
function renderScoreCard(data) {
  const pct = data.similarity_percent;
  const display = `${pct.toFixed(1)}%`;

  el.scoreMain.textContent = display;

  // Colour level
  const card = el.scoreCard;
  card.classList.remove('level-low', 'level-mid', 'level-high');

  let verdict;
  if (pct < 20) {
    card.classList.add('level-low');
    verdict = 'Likely Original';
  } else if (pct < 50) {
    card.classList.add('level-mid');
    verdict = 'Moderate Similarity';
  } else {
    card.classList.add('level-high');
    verdict = 'High Similarity';
  }

  el.scoreVerdict.textContent = verdict;

  // Animate bar (defer so CSS transition fires)
  requestAnimationFrame(() => {
    el.scoreBar.style.width = `${Math.min(pct, 100)}%`;
  });
}

/* ─── Breakdown ──────────────────────────────────────────── */
function renderBreakdown(data) {
  setMetricBar(el.barJaccard, el.valJaccard, data.jaccard_score);
  setMetricBar(el.barTfidf,   el.valTfidf,   data.tfidf_score);

  if (data.semantic_available) {
    el.rowSemantic.hidden = false;
    setMetricBar(el.barSemantic, el.valSemantic, data.semantic_score);
    el.metaSemStatus.textContent = '✓ Semantic model active';
  } else {
    el.rowSemantic.hidden = true;
    el.metaSemStatus.textContent = '⚠ Semantic model not installed';
  }

  el.metaTime.textContent = `⏱ ${data.processing_time_ms} ms`;
}

function setMetricBar(barEl, valEl, score) {
  const pct = Math.round(score * 100);
  valEl.textContent = `${pct}%`;
  requestAnimationFrame(() => { barEl.style.width = `${pct}%`; });
}

/* ─── Matched Pairs ──────────────────────────────────────── */
function renderMatches(pairs) {
  el.matchesList.innerHTML = '';
  el.matchCount.textContent = pairs.length;

  if (!pairs.length) {
    el.noMatches.hidden = false;
    return;
  }

  el.noMatches.hidden = true;

  pairs.forEach((pair, idx) => {
    const pct = Math.round(pair.score * 100);
    const tier = pct >= 70 ? 'high' : pct >= 40 ? 'mid' : 'low';

    const card = document.createElement('div');
    card.className = 'match-card';
    card.dataset.method = pair.method;
    card.innerHTML = `
      <div class="match-card-header">
        <span class="match-score-pill pill-${tier}">${pct}% match</span>
        <span class="match-method">${pair.method}</span>
      </div>
      <div class="match-body">
        <div class="match-col">${escHtml(pair.sentence_a)}</div>
        <div class="match-col-sep"></div>
        <div class="match-col">${escHtml(pair.sentence_b)}</div>
      </div>
    `;
    el.matchesList.appendChild(card);
  });
}

/* ─── Semantic-only filter ───────────────────────────────── */
el.toggleSemOnly.addEventListener('change', () => {
  const semanticOnly = el.toggleSemOnly.checked;
  const filtered = semanticOnly
    ? currentMatchedPairs.filter(p => p.method === 'semantic')
    : currentMatchedPairs;
  renderMatches(filtered);
});

/* ─── Side-by-side highlights ────────────────────────────── */

/**
 * Given original texts and matched pairs, mark matched sentences in each pane.
 * Each matched pair gets a colour class (high/mid/low) by score tier.
 */
function renderHighlights(textA, textB, pairs) {
  const matchedA = new Map(); // sentence_text → tier
  const matchedB = new Map();

  pairs.forEach(pair => {
    const pct  = pair.score * 100;
    const tier = pct >= 70 ? 'high' : pct >= 40 ? 'mid' : 'low';
    matchedA.set(pair.sentence_a.trim(), tier);
    matchedB.set(pair.sentence_b.trim(), tier);
  });

  el.paneA.innerHTML = highlightText(textA, matchedA);
  el.paneB.innerHTML = highlightText(textB, matchedB);
}

/**
 * Split text into sentences (naive regex; mirrors NLTK sentence splits closely
 * enough for display purposes) and wrap matched ones in <mark> spans.
 */
function highlightText(text, matchMap) {
  const sentences = text.split(/(?<=[.!?])\s+/);

  const parts = sentences.map(sentence => {
    const cleaned = sentence.trim().toLowerCase();

    let matchedTier = null;

    for (const [key, tier] of matchMap.entries()) {
      const k = key.toLowerCase();

      // 🔥 STRONG fuzzy match (word overlap)
      const wordsA = new Set(cleaned.split(/\s+/));
      const wordsB = new Set(k.split(/\s+/));

      let common = 0;
      wordsA.forEach(w => {
        if (wordsB.has(w)) common++;
      });

      const overlap = common / Math.max(wordsA.size, wordsB.size);

      if (overlap > 0.6) {   // 🔥 THIS IS KEY
        matchedTier = tier;
        break;
      }
    }

    if (matchedTier) {
      return `<mark class="hl hl-${matchedTier}">${escHtml(sentence)}</mark>`;
    }

    return escHtml(sentence);
  });

  return parts.join(' ');
}

/* ══════════════════════════════════════════════
   RESET
══════════════════════════════════════════════ */

function hideResults() {
  el.resultsSection.hidden = true;
  el.scoreBar.style.width = '0%';
  [el.barJaccard, el.barTfidf, el.barSemantic].forEach(b => (b.style.width = '0%'));
  currentMatchedPairs = [];
}

el.btnReset.addEventListener('click', () => {
  hideResults();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* ══════════════════════════════════════════════
   UTILITIES
══════════════════════════════════════════════ */

/** Escape HTML special characters to prevent XSS. */
function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/* ══════════════════════════════════════════════
   INITIALISE
══════════════════════════════════════════════ */

checkHealth();
updateCount(el.textA, el.countA);
updateCount(el.textB, el.countB);
