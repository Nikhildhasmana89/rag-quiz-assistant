/**
 * EvidenceRAG — Frontend Application Controller
 * Handles view routing, state persistence, API integration, and responsive DOM rendering.
 */

// Global State
const state = {
  activeView: 'dashboard',
  queriesCount: parseInt(localStorage.getItem('rag_queries_count') || '0', 10),
  quizzesCount: parseInt(localStorage.getItem('rag_quizzes_count') || '0', 10),
  indexedDocument: JSON.parse(localStorage.getItem('rag_active_document') || 'null'),
  quizQuestions: [],
  lastLatencyMs: null,
};

// DOM Selectors Helper
const $ = (id) => document.getElementById(id);

// Toast Notification Engine
function showToast(message, type = 'info') {
  const container = $('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '⚠️' : 'ℹ️'}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(-10px)';
    toast.style.transition = 'all 0.25s ease';
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

// API Fetch Helper
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Request failed with status ${response.status}`);
  }
  return body;
}

// View Routing Switcher
function switchView(viewName) {
  state.activeView = viewName;

  // Toggle Nav Items
  document.querySelectorAll('.sidebar-nav .nav-item').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === viewName);
  });

  // Toggle View Panels
  document.querySelectorAll('.view-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `view-${viewName}`);
  });

  // Update Breadcrumbs
  const titles = {
    dashboard: 'Dashboard',
    documents: 'My Documents',
    ask: 'Ask AI',
    compare: 'Compare Documents',
    quiz: 'Generate Quiz',
    evaluation: 'Research Evaluation',
  };
  $('current-view-title').textContent = titles[viewName] || 'Dashboard';

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Make switchView globally accessible
window.switchView = switchView;

// Suggested Question Helper
function useSuggestedQuery(text) {
  switchView('ask');
  const questionEl = $('question');
  if (questionEl) {
    questionEl.value = text;
    questionEl.focus();
  }
}
window.useSuggestedQuery = useSuggestedQuery;

// Render Pipeline Health & Metadata
function renderStatus(status) {
  // Sidebar Health
  $('api-health').textContent = 'online';
  $('api-health').className = 'badge badge-online';

  const hasStoreError = Boolean(status.collection_info?.error);
  const storeBadge = $('store-health');
  storeBadge.textContent = hasStoreError ? 'not ready' : 'ready';
  storeBadge.className = `badge ${hasStoreError ? '' : 'badge-emerald'}`;

  const chunkCount = status.collection_info?.document_count ?? 0;
  $('document-count').textContent = chunkCount;

  // Sidebar Specs
  $('sidebar-model').textContent = status.llm_model || '--';
  $('sidebar-embedding').textContent = status.collection_info?.embedding_model || status.embedding_model || 'all-MiniLM-L6-v2';

  // Topbar
  $('provider-badge').textContent = (status.llm_provider || '--').toUpperCase();
  $('connection-label').textContent = 'Pipeline Active';
  $('live-indicator').className = 'chip-dot live';

  // Dashboard KPIs
  $('kpi-chunk-count').textContent = chunkCount;
  $('kpi-queries-count').textContent = state.queriesCount;
  $('kpi-quiz-count').textContent = state.quizzesCount;

  // Evaluation Specs
  $('eval-provider').textContent = status.llm_provider || '--';
  $('eval-model').textContent = status.llm_model || '--';
  $('eval-embedding').textContent = status.collection_info?.embedding_model || 'all-MiniLM-L6-v2';
  $('eval-chunk-size').textContent = `${status.chunk_size ?? 500} characters`;
  $('eval-chunk-overlap').textContent = `${status.chunk_overlap ?? 0} characters`;

  // Render Document Library Cards
  renderDocumentLibrary(chunkCount);
}

// Render Document Library
function renderDocumentLibrary(chunkCount) {
  const docCard = $('active-doc-card');
  const emptyState = $('documents-empty');
  const navDocCount = $('nav-doc-count');
  const kpiDocCount = $('kpi-doc-count');
  const kpiDocSub = $('kpi-doc-sub');
  const cmpDocName = $('cmp-doc-name');

  if (chunkCount > 0) {
    const docInfo = state.indexedDocument || {
      name: 'Uploaded Research Document.pdf',
      pages: 'Available',
      chunks: chunkCount,
      size: 'Local ChromaDB Store',
    };

    if (docCard) {
      docCard.style.display = 'flex';
      $('doc-item-name').textContent = docInfo.name;
      $('doc-item-pages').textContent = typeof docInfo.pages === 'number' ? `${docInfo.pages} pages` : `${docInfo.pages}`;
      $('doc-item-chunks').textContent = `${chunkCount} vector chunks`;
    }

    if (emptyState) emptyState.style.display = 'none';
    if (navDocCount) navDocCount.textContent = '1';
    if (kpiDocCount) kpiDocCount.textContent = '1';
    if (kpiDocSub) kpiDocSub.textContent = docInfo.name;
    if (cmpDocName) cmpDocName.textContent = docInfo.name;
  } else {
    if (docCard) docCard.style.display = 'none';
    if (emptyState) emptyState.style.display = 'block';
    if (navDocCount) navDocCount.textContent = '0';
    if (kpiDocCount) kpiDocCount.textContent = '0';
    if (kpiDocSub) kpiDocSub.textContent = 'No PDF indexed';
    if (cmpDocName) cmpDocName.textContent = 'None currently indexed';
  }
}

// Refresh Status from Server
async function refreshStatus() {
  try {
    const data = await api('/api/status');
    renderStatus(data);
  } catch (error) {
    $('api-health').textContent = 'offline';
    $('api-health').className = 'badge';
    $('store-health').textContent = 'offline';
    $('connection-label').textContent = 'Disconnected';
    $('live-indicator').className = 'chip-dot';
    showToast(`Server status check failed: ${error.message}`, 'error');
  }
}

// Setup Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  // Navigation Sidebar clicks
  document.querySelectorAll('.sidebar-nav .nav-item').forEach((button) => {
    button.addEventListener('click', () => {
      const view = button.dataset.view;
      if (view) switchView(view);
    });
  });

  // Topbar quick upload button
  $('topbar-upload-btn')?.addEventListener('click', () => switchView('documents'));

  // Refresh Status button
  $('refresh-status')?.addEventListener('click', () => {
    refreshStatus();
    showToast('Checking pipeline status...', 'info');
  });

  // File picker change listener
  $('pdf-file')?.addEventListener('change', (event) => {
    const file = event.target.files[0];
    $('file-label').textContent = file ? file.name : 'Choose a PDF file';
  });

  // Dropzone drag-and-drop
  const dropzone = $('dropzone');
  if (dropzone) {
    ['dragenter', 'dragover'].forEach((name) => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragging');
      });
    });

    ['dragleave', 'drop'].forEach((name) => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragging');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length) {
        $('pdf-file').files = e.dataTransfer.files;
        $('file-label').textContent = e.dataTransfer.files[0].name;
      }
    });
  }

  // Clear query button
  $('clear-query-btn')?.addEventListener('click', () => {
    $('question').value = '';
    $('question').focus();
  });

  // Copy answer button
  $('copy-answer-btn')?.addEventListener('click', () => {
    const answerText = $('answer').textContent;
    if (answerText) {
      navigator.clipboard.writeText(answerText);
      showToast('Answer copied to clipboard!', 'success');
    }
  });

  // --------------------------------------------------------------------------
  // Document Ingestion Form
  // --------------------------------------------------------------------------
  $('upload-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = $('pdf-file').files[0];
    const resultBanner = $('ingest-result');
    const submitBtn = $('upload-submit');
    const processingState = $('processing-state');
    const processingStep = $('processing-step');
    const processingDetail = $('processing-detail');

    if (!file) {
      resultBanner.textContent = 'Please select a PDF file first.';
      resultBanner.className = 'result-banner error';
      return;
    }

    // UI Loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('span').textContent = 'Ingesting...';
    processingState.style.display = 'flex';
    resultBanner.textContent = '';
    resultBanner.className = 'result-banner';

    processingStep.textContent = 'Extracting text...';
    processingDetail.textContent = `Reading ${file.name} with pdfplumber`;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('recreate', $('recreate').checked);

    try {
      setTimeout(() => {
        if (submitBtn.disabled) {
          processingStep.textContent = 'Chunking & Embedding...';
          processingDetail.textContent = 'Creating 500-char chunks and vectorizing via all-MiniLM-L6-v2';
        }
      }, 1200);

      const response = await api('/api/ingest', {
        method: 'POST',
        body: formData,
      });

      processingState.style.display = 'none';
      resultBanner.textContent = `Indexed ${response.chunks_created} chunks from ${response.pages_extracted} pages into collection "${response.collection_name}".`;
      resultBanner.className = 'result-banner success';
      showToast('PDF successfully indexed into ChromaDB!', 'success');

      // Persist active document info in localStorage
      state.indexedDocument = {
        name: file.name,
        size: `${(file.size / 1024).toFixed(1)} KB`,
        pages: response.pages_extracted,
        chunks: response.chunks_created,
        timestamp: new Date().toLocaleTimeString(),
      };
      localStorage.setItem('rag_active_document', JSON.stringify(state.indexedDocument));

      await refreshStatus();
    } catch (err) {
      processingState.style.display = 'none';
      resultBanner.textContent = err.message;
      resultBanner.className = 'result-banner error';
      showToast(`Ingestion failed: ${err.message}`, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector('span').textContent = 'Ingest & Index Document';
    }
  });

  // --------------------------------------------------------------------------
  // RAG Query / Ask AI Form
  // --------------------------------------------------------------------------
  $('query-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const queryInput = $('question');
    const questionText = queryInput.value.trim();
    const submitBtn = $('ask-submit');
    const answerEl = $('answer');
    const answerPlaceholder = $('answer-placeholder');
    const sourcesEl = $('sources');
    const evidenceSection = $('evidence-section');
    const evidenceStatus = $('evidence-status');
    const citationsSection = $('citations-section');
    const citationsList = $('citations-list');
    const citationsCountBadge = $('citations-count-badge');
    const latencyBadge = $('latency-badge');
    const copyBtn = $('copy-answer-btn');
    const countBadge = $('evidence-count-badge');
    const evalLatency = $('eval-live-latency');

    if (!questionText) {
      showToast('Please type a question first.', 'info');
      return;
    }

    // UI Loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('span').textContent = 'Searching...';
    if (answerPlaceholder) answerPlaceholder.style.display = 'none';
    answerEl.className = 'answer-body loading';
    answerEl.textContent = 'Searching vector index for relevant passages and synthesizing answer...';
    evidenceSection.style.display = 'none';
    evidenceStatus.style.display = 'none';
    if (citationsSection) citationsSection.style.display = 'none';
    latencyBadge.style.display = 'none';
    copyBtn.style.display = 'none';

    const startTime = performance.now();

    try {
      const payload = {
        query: questionText,
        top_k: Math.max(1, Math.min(20, parseInt($('top-k').value || '5', 10))),
      };

      const response = await api('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const latencyMs = Math.round(performance.now() - startTime);
      state.lastLatencyMs = latencyMs;
      const latencySec = (latencyMs / 1000).toFixed(2);

      // Render Answer with inline citation markers
      answerEl.className = 'answer-body';
      if (response.annotated_response) {
        const safeText = response.annotated_response
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/\[(\d+)\]/g, '<span class="citation-marker">[$1]</span>');
        answerEl.innerHTML = safeText;
      } else {
        answerEl.textContent = response.response;
      }

      // Verification Status Badge
      if (response.verification && response.verification.status) {
        const vStatus = response.verification.status;
        const statusConfig = {
          supported: { text: '✓ Supported by Evidence', cls: 'status-supported' },
          partially_supported: { text: '⚠ Partially Supported', cls: 'status-partially_supported' },
          contradicted: { text: '✕ Contradicted by Evidence', cls: 'status-contradicted' },
          insufficient_evidence: { text: '? Insufficient Evidence', cls: 'status-insufficient_evidence' },
        };
        const cfg = statusConfig[vStatus] || { text: `Status: ${vStatus}`, cls: 'status-supported' };
        evidenceStatus.textContent = cfg.text;
        evidenceStatus.className = `evidence-badge ${cfg.cls}`;
        evidenceStatus.style.display = 'inline-block';
      } else {
        evidenceStatus.textContent = '✓ Answer Synthesized';
        evidenceStatus.className = 'evidence-badge status-supported';
        evidenceStatus.style.display = 'inline-block';
      }
      latencyBadge.textContent = `⚡ ${latencySec}s`;
      latencyBadge.style.display = 'inline-block';
      copyBtn.style.display = 'inline-block';

      if (evalLatency) {
        evalLatency.textContent = `${latencyMs} ms`;
      }

      // Render Grounded Citations & Provenance (Step 6)
      if (citationsSection && citationsList) {
        citationsList.innerHTML = '';
        const hasCitations = response.citations && response.citations.length > 0;
        const hasUnsupported = response.unsupported_claims && response.unsupported_claims.length > 0;

        if (hasCitations || hasUnsupported) {
          citationsSection.style.display = 'block';
          if (citationsCountBadge) {
            citationsCountBadge.textContent = `${(response.citations || []).length} citation${(response.citations || []).length === 1 ? '' : 's'}`;
          }

          if (hasCitations) {
            response.citations.forEach((cit) => {
              const card = document.createElement('div');
              const statusClass = cit.claim_status === 'supported' ? 'supported' : (cit.claim_status === 'partially_supported' ? 'partially-supported' : 'contradicted');
              card.className = `citation-card ${statusClass}`;
              const scoreText = cit.reranker_score !== null && cit.reranker_score !== undefined
                ? `Rerank: ${Number(cit.reranker_score).toFixed(3)}`
                : `Score: ${Number(cit.confidence_score || 1.0).toFixed(2)}`;
              
              const secPill = cit.section ? `<span class="citation-pill">Sec: ${cit.section}</span>` : '';
              card.innerHTML = `
                <div class="citation-top-row">
                  <span class="citation-claim-title">[${cit.citation_index}] "${cit.claim}"</span>
                  <span class="evidence-badge ${statusClass === 'supported' ? 'status-supported' : (statusClass === 'partially-supported' ? 'status-partially_supported' : 'status-contradicted')}">
                    ${cit.claim_status.toUpperCase()}
                  </span>
                </div>
                <div class="citation-meta-pills">
                  <span class="citation-pill doc">📄 ${cit.document_name}</span>
                  <span class="citation-pill">Page ${cit.page_number}</span>
                  ${secPill}
                  <span class="citation-pill">Chunk: ${cit.chunk_id}</span>
                  <span class="citation-pill score">${scoreText}</span>
                </div>
                ${cit.evidence_text ? `<div class="citation-snippet">"${cit.evidence_text}"</div>` : ''}
              `;
              citationsList.appendChild(card);
            });
          }

          if (hasUnsupported) {
            const unBanner = document.createElement('div');
            unBanner.className = 'unsupported-banner';
            const unList = response.unsupported_claims.map((u) => `• "${u}" (Zero fake citations generated)`).join('<br>');
            unBanner.innerHTML = `<strong>⚠️ Ungrounded Statements (${response.unsupported_claims.length}):</strong><br>${unList}`;
            citationsList.appendChild(unBanner);
          }
        } else {
          citationsSection.style.display = 'none';
        }
      }

      // Render Retrieved Evidence Chunks with Provenance Metadata
      sourcesEl.innerHTML = '';
      const rawChunks = response.retrieved_chunks || [];
      if (rawChunks.length > 0) {
        evidenceSection.style.display = 'block';
        countBadge.textContent = `${rawChunks.length} chunks retrieved`;

        rawChunks.forEach((chunk, idx) => {
          const item = document.createElement('div');
          item.className = 'source-item';
          const meta = chunk.metadata || {};
          const docName = meta.source_file || chunk.source_file || 'Document';
          const pageNum = meta.page_number || chunk.page_number || 1;
          const secName = meta.section || chunk.section || '';
          const cid = chunk.chunk_id || chunk.id || `chunk_${idx + 1}`;
          const rerankScore = chunk.reranker_score;
          const scoreInfo = rerankScore !== undefined && rerankScore !== null
            ? `Rerank: ${Number(rerankScore).toFixed(3)}`
            : `Score: ${Number(chunk.hybrid_score || chunk.dense_score || 1.0).toFixed(2)}`;

          item.innerHTML = `
            <div class="source-meta">
              <span class="source-tag">Chunk #${idx + 1} (${cid})</span>
              <span class="source-length">📄 ${docName} · P.${pageNum}${secName ? ` · ${secName}` : ''} · ${scoreInfo}</span>
            </div>
            <div class="source-content">${chunk.text || ''}</div>
          `;
          sourcesEl.appendChild(item);
        });
      } else if (response.retrieved_documents && response.retrieved_documents.length) {
        evidenceSection.style.display = 'block';
        countBadge.textContent = `${response.retrieved_documents.length} chunks retrieved`;

        response.retrieved_documents.forEach((docText, idx) => {
          const item = document.createElement('div');
          item.className = 'source-item';
          item.innerHTML = `
            <div class="source-meta">
              <span class="source-tag">Evidence Chunk #${idx + 1}</span>
              <span class="source-length">${docText.length} chars · Relevance: High</span>
            </div>
            <div class="source-content">${docText}</div>
          `;
          sourcesEl.appendChild(item);
        });
      }

      // Increment queries counter
      state.queriesCount += 1;
      localStorage.setItem('rag_queries_count', state.queriesCount.toString());
      $('kpi-queries-count').textContent = state.queriesCount;

      showToast('Answer synthesized with grounded evidence.', 'success');
    } catch (err) {
      answerEl.className = 'answer-body';
      answerEl.textContent = `Error: ${err.message}`;
      showToast(`Query failed: ${err.message}`, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector('span').textContent = 'Ask the Index';
    }
  });

  // --------------------------------------------------------------------------
  // Practice Quiz Generation & Evaluation
  // --------------------------------------------------------------------------
  $('quiz-generate')?.addEventListener('click', async () => {
    const generateBtn = $('quiz-generate');
    const statusEl = $('quiz-status');
    const listEl = $('quiz-list');
    const emptyState = $('quiz-empty');

    generateBtn.disabled = true;
    generateBtn.querySelector('span').textContent = 'Generating...';
    statusEl.textContent = 'Reading indexed passages and deriving technical questions...';
    statusEl.className = 'result-banner';
    listEl.innerHTML = '';

    try {
      const count = Math.max(1, Math.min(15, parseInt($('quiz-count').value || '5', 10)));
      const response = await api('/api/quiz/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_questions: count }),
      });

      state.quizQuestions = response.questions || [];

      if (!state.quizQuestions.length) {
        statusEl.textContent = 'No questions could be generated. Make sure a PDF is indexed first.';
        statusEl.className = 'result-banner error';
        return;
      }

      statusEl.textContent = `Generated ${state.quizQuestions.length} interview questions grounded in the indexed document.`;
      statusEl.className = 'result-banner success';
      if (emptyState) emptyState.style.display = 'none';

      renderQuizCards();
      showToast(`Generated ${state.quizQuestions.length} practice questions!`, 'success');
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.className = 'result-banner error';
      showToast(`Quiz generation failed: ${err.message}`, 'error');
    } finally {
      generateBtn.disabled = false;
      generateBtn.querySelector('span').textContent = 'Generate Quiz';
    }
  });

  function renderQuizCards() {
    const listEl = $('quiz-list');
    listEl.innerHTML = '';

    state.quizQuestions.forEach((q, idx) => {
      const card = document.createElement('div');
      card.className = 'quiz-item-card';
      card.dataset.id = q.id;

      card.innerHTML = `
        <div class="quiz-question-header">
          <span class="quiz-badge">Question ${idx + 1} of ${state.quizQuestions.length}</span>
        </div>
        <h4 class="quiz-question-title">${q.question}</h4>
        <textarea class="quiz-answer-input" rows="3" placeholder="Type your technical explanation here..."></textarea>
        <div class="quiz-card-footer">
          <button class="btn-primary-sm quiz-eval-btn" type="button">
            <span>Submit & Grade Answer</span>
            <span class="btn-arrow">↗</span>
          </button>
        </div>
        <div class="quiz-feedback-box" style="display:none;"></div>
      `;

      const evalBtn = card.querySelector('.quiz-eval-btn');
      const textarea = card.querySelector('.quiz-answer-input');
      const feedbackBox = card.querySelector('.quiz-feedback-box');

      evalBtn.addEventListener('click', async () => {
        const userAnswer = textarea.value.trim();
        if (!userAnswer) {
          showToast('Please type an answer before submitting.', 'info');
          return;
        }

        evalBtn.disabled = true;
        evalBtn.querySelector('span').textContent = 'Grading...';
        feedbackBox.style.display = 'none';

        try {
          const evalResult = await api('/api/quiz/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              question: q.question,
              expected_answer: q.expected_answer,
              context: q.context,
              user_answer: userAnswer,
            }),
          });

          const verdictSlug = (evalResult.verdict || 'unknown').toLowerCase().replace(/\s+/g, '-');
          feedbackBox.className = `quiz-feedback-box verdict-${verdictSlug}`;
          feedbackBox.style.display = 'flex';
          feedbackBox.innerHTML = `
            <div class="quiz-feedback-score">Verdict: ${evalResult.verdict} (${evalResult.score}/100)</div>
            <div class="quiz-feedback-desc">${evalResult.feedback}</div>
          `;

          // Increment quizzes evaluated counter
          state.quizzesCount += 1;
          localStorage.setItem('rag_quizzes_count', state.quizzesCount.toString());
          $('kpi-quiz-count').textContent = state.quizzesCount;

          showToast(`Answer graded: ${evalResult.verdict} (${evalResult.score}/100)`, 'info');
        } catch (err) {
          feedbackBox.className = 'quiz-feedback-box verdict-incorrect';
          feedbackBox.style.display = 'flex';
          feedbackBox.innerHTML = `<div class="quiz-feedback-desc">Grading error: ${err.message}</div>`;
          showToast(`Evaluation failed: ${err.message}`, 'error');
        } finally {
          evalBtn.disabled = false;
          evalBtn.querySelector('span').textContent = 'Submit & Grade Answer';
        }
      });

      listEl.appendChild(card);
    });
  }

  // Initial status fetch on boot
  refreshStatus();
});
