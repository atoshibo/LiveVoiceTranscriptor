const LS_KEY = 'lvt_token';
let TOKEN = localStorage.getItem(LS_KEY) || '';
let currentSession = null;
let currentSessionTab = 'status';
let uploadQueue = [];
let uploadQueueCounter = 0;
let uploadQueueProcessing = false;
let uploadQueuePollHandle = null;

function api(path, opts = {}) {
  const headers = { Authorization: 'Bearer ' + TOKEN, ...(opts.headers || {}) };
  return fetch(path, { ...opts, headers }).then((response) => {
    if (response.status === 401 || response.status === 403) {
      showAuthGate();
      throw new Error('auth');
    }
    return response;
  });
}

function apiJson(path, opts) {
  return api(path, opts).then(async (response) => {
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body?.detail === 'string'
        ? body.detail
        : body?.detail?.message || body?.message || `HTTP ${response.status}`;
      const error = new Error(detail);
      error.status = response.status;
      error.body = body;
      throw error;
    }
    return body;
  });
}

function escHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function badgeClass(state) {
  const normalized = String(state || 'unknown').toLowerCase();
  if (['done', 'success', 'stable', 'healthy'].includes(normalized)) return 'done';
  if (['error', 'failed', 'degraded', 'unhealthy', 'cancelled'].includes(normalized)) return 'error';
  if (['queued', 'pending', 'uploaded', 'created', 'waiting'].includes(normalized)) return 'queued';
  return 'running';
}

function renderBadge(state, label) {
  const text = label || state || 'unknown';
  return `<span class="badge ${badgeClass(state)}">${escHtml(text)}</span>`;
}

function formatDateTime(value) {
  if (!value) return 'N/A';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return escHtml(value);
  return escHtml(date.toLocaleString());
}

function formatDurationMs(value) {
  if (value === null || value === undefined) return 'N/A';
  const totalMs = Number(value) || 0;
  if (totalMs < 1000) return `${totalMs} ms`;
  const totalSec = Math.round(totalMs / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h) return `${h}h ${m}m ${s}s`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function yesNo(value) {
  return value ? 'Yes' : 'No';
}

function joinValues(values, fallback = 'None') {
  return values && values.length ? values.join(', ') : fallback;
}

function renderReasons(reasons) {
  const list = (reasons || []).filter(Boolean);
  if (!list.length) return '<span class="muted">None</span>';
  return list.map((item) => `<span class="pill">${escHtml(item)}</span>`).join(' ');
}

function setActiveSessionTab(tab) {
  document.querySelectorAll('.tab-bar button[data-session-tab]').forEach((button) => {
    button.classList.toggle('active', button.dataset.sessionTab === tab);
  });
}

function submitToken() {
  TOKEN = document.getElementById('token-input').value.trim();
  if (!TOKEN) {
    document.getElementById('auth-error').textContent = 'Token required';
    return;
  }
  apiJson('/api/v2/health').then(() => {
    localStorage.setItem(LS_KEY, TOKEN);
    document.getElementById('auth-gate').classList.add('hidden');
    showPage('dashboard');
  }).catch(() => {
    document.getElementById('auth-error').textContent = 'Invalid token or server unreachable';
  });
}

function showAuthGate() {
  document.querySelectorAll('[id^="page-"]').forEach((page) => page.classList.add('hidden'));
  document.getElementById('auth-gate').classList.remove('hidden');
}

function logout() {
  TOKEN = '';
  localStorage.removeItem(LS_KEY);
  showAuthGate();
}

function showPage(page) {
  document.querySelectorAll('[id^="page-"]').forEach((node) => node.classList.add('hidden'));
  document.querySelectorAll('.nav a').forEach((node) => node.classList.remove('active'));
  const el = document.getElementById('page-' + page);
  if (el) el.classList.remove('hidden');
  const nav = document.getElementById('nav-' + page);
  if (nav) nav.classList.add('active');

  if (page === 'dashboard') loadDashboard();
  else if (page === 'sessions') loadSessions();
  else if (page === 'upload') {
    onUploadSelectionChanged();
    renderUploadQueue();
  } else if (page === 'models') loadModels();
  else if (page === 'diagnostics') loadDiagnostics();
}

function loadDashboard() {
  apiJson('/api/v2/health').then((health) => {
    document.getElementById('health-info').innerHTML = `
      <div class="kv"><span class="k">Version</span><span class="v">${escHtml(health.version)}</span></div>
      <div class="kv"><span class="k">TLS</span><span class="v">${yesNo(health.tls_enabled)}</span></div>
      <div class="kv"><span class="k">GPU</span><span class="v">${health.gpu_available ? 'Available' : 'Not available'}${health.gpu_reason ? ` (${escHtml(health.gpu_reason)})` : ''}</span></div>
      <div class="kv"><span class="k">Device</span><span class="v">${escHtml(health.selected_device || 'cpu')}</span></div>
      <div class="kv"><span class="k">Compute</span><span class="v">${escHtml(health.selected_compute_type || 'int8')}</span></div>
      <div class="kv"><span class="k">Models Dir</span><span class="v mono">${escHtml(health.models_dir || '?')}</span></div>`;
  }).catch(() => {
    document.getElementById('health-info').textContent = 'Failed to load';
  });

  apiJson('/api/v2/system/gpu').then((gpu) => {
    document.getElementById('gpu-info').innerHTML = `
      <div class="kv"><span class="k">GPU Name</span><span class="v">${escHtml(gpu.gpu_name || 'N/A')}</span></div>
      <div class="kv"><span class="k">Memory</span><span class="v">${escHtml(gpu.memory_used_mb ?? '?')}/${escHtml(gpu.memory_total_mb ?? '?')} MB</span></div>
      <div class="kv"><span class="k">Main Queue</span><span class="v">${escHtml(gpu.queue_depth || 0)} jobs</span></div>
      <div class="kv"><span class="k">Partial Queue</span><span class="v">${escHtml(gpu.partial_queue_depth || 0)} jobs</span></div>
      <div class="kv"><span class="k">Worker Started</span><span class="v">${formatDateTime(gpu.worker_started_at)}</span></div>`;
  }).catch(() => {
    document.getElementById('gpu-info').textContent = 'Failed to load';
  });

  apiJson('/api/sessions').then((payload) => {
    const sessions = payload.sessions || [];
    if (!sessions.length) {
      document.getElementById('recent-sessions').textContent = 'No sessions yet.';
      return;
    }
    let html = '<table><tr><th>Session</th><th>Status</th><th>Mode</th><th>Created</th><th></th></tr>';
    sessions.slice(0, 10).forEach((session) => {
      html += `<tr><td class="mono">${escHtml(session.session_id.slice(0, 8))}...</td>
        <td>${renderBadge(session.status)}</td>
        <td>${escHtml(session.mode || '')}</td>
        <td>${formatDateTime(session.created_at)}</td>
        <td><a onclick="openSession('${session.session_id}')">View</a></td></tr>`;
    });
    html += '</table>';
    document.getElementById('recent-sessions').innerHTML = html;
  }).catch(() => {
    document.getElementById('recent-sessions').textContent = 'Failed';
  });
}

function loadSessions() {
  apiJson('/api/v2/sessions/grouped').then((payload) => {
    let html = '';
    (payload.groups || []).forEach((group) => {
      html += `<h3 style="margin:16px 0 8px">${escHtml(group.label)} (${group.sessions.length})</h3>`;
      html += '<table><tr><th>Session</th><th>State</th><th>Backend</th><th>Mode</th><th>Chunks</th><th>Transcript</th><th>Created</th><th></th></tr>';
      group.sessions.forEach((session) => {
        html += `<tr><td class="mono">${escHtml(session.session_id.slice(0, 8))}...</td>
          <td>${renderBadge(session.state)}</td>
          <td>${escHtml(session.backend_outcome || '')}</td>
          <td>${escHtml(session.mode || '')}</td>
          <td>${escHtml(session.chunk_count || 0)}</td>
          <td>${yesNo(session.transcript_present)}</td>
          <td>${formatDateTime(session.created_at)}</td>
          <td><a onclick="openSession('${session.session_id}')">View</a>
              <a onclick="deleteSession('${session.session_id}')" style="color:var(--red);margin-left:8px">Del</a></td></tr>`;
      });
      html += '</table>';
    });
    if (!html) html = '<p>No sessions.</p>';
    html += `<p style="margin-top:12px;color:var(--text-dim)">Total: ${escHtml(payload.total_sessions || 0)}</p>`;
    document.getElementById('sessions-grouped').innerHTML = html;
  }).catch((error) => {
    document.getElementById('sessions-grouped').innerHTML = `<p style="color:var(--red)">Failed to load sessions: ${escHtml(error.message)}</p>`;
  });
}

function openSession(sessionId) {
  currentSession = sessionId;
  currentSessionTab = 'status';
  document.querySelectorAll('[id^="page-"]').forEach((page) => page.classList.add('hidden'));
  document.getElementById('page-session-detail').classList.remove('hidden');
  loadSessionDetail(sessionId);
}

async function loadSessionDetail(sessionId) {
  try {
    const [job, detail] = await Promise.all([
      apiJson(`/api/v2/jobs/${sessionId}`),
      apiJson(`/api/v2/sessions/${sessionId}`),
    ]);
    const diarization = job.diarization || detail.diarization || {};
    const qualityGate = job.quality_gate || detail.quality_gate || {};
    const contextSpanCount = job.context_span_count ?? detail.context_span_count ?? 0;
    document.getElementById('session-header').innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap">
        <div class="stack" style="gap:8px">
          <div><h2 class="mono">${escHtml(sessionId)}</h2></div>
          <div>${renderBadge(job.state)} <span class="muted" style="margin-left:8px">${escHtml(job.backend_outcome || '')}</span></div>
          <div class="small muted">${job.failure_category ? `Failure category: ${escHtml(job.failure_category)}` : ''}</div>
        </div>
        <div>
          <button class="secondary" onclick="showPage('sessions')">Back</button>
          ${job.state === 'error' ? `<button onclick="retrySession('${sessionId}')" style="margin-left:8px">Retry</button>` : ''}
          <button class="danger" onclick="deleteSession('${sessionId}')" style="margin-left:8px">Delete</button>
        </div>
      </div>
      <div class="status-bar" style="margin-top:12px">
        <div class="progress-bar"><div class="progress-fill" style="width:${escHtml(job.progress?.processing || 0)}%"></div></div>
        <span style="font-size:12px">${escHtml(job.progress?.processing || 0)}% - ${escHtml(job.progress?.stage || 'pending')}</span>
      </div>
      <div class="grid" style="margin-top:8px">
        <div>
          <div class="kv"><span class="k">Queued</span><span class="v">${formatDateTime(job.queued_at)}</span></div>
          <div class="kv"><span class="k">Started</span><span class="v">${formatDateTime(job.started_at)}</span></div>
          <div class="kv"><span class="k">Finished</span><span class="v">${formatDateTime(job.finished_at)}</span></div>
          <div class="kv"><span class="k">Context Spans</span><span class="v">${escHtml(contextSpanCount)}</span></div>
        </div>
        <div>
          <div class="kv"><span class="k">Model</span><span class="v">${escHtml(job.meta?.model_size || detail.model_size || 'auto')}</span></div>
          <div class="kv"><span class="k">Language</span><span class="v">${escHtml(job.meta?.language_requested || detail.language || 'auto')}</span></div>
          <div class="kv"><span class="k">Diarization</span><span class="v">${escHtml(diarization.policy || 'auto')} / ${escHtml(diarization.status || 'unknown')}</span></div>
          <div class="kv"><span class="k">Quality Gate</span><span class="v">${escHtml(qualityGate.session_quality_status || 'unknown')}</span></div>
        </div>
      </div>`;
    showSessionTab(currentSessionTab);
  } catch (error) {
    document.getElementById('session-header').innerHTML = `<p style="color:var(--red)">Failed to load session: ${escHtml(error.message)}</p>`;
  }
}

async function showSessionTab(tab) {
  currentSessionTab = tab;
  setActiveSessionTab(tab);
  const sessionId = currentSession;
  if (!sessionId) return;

  const el = document.getElementById('session-tab-content');
  try {
    if (tab === 'status') {
      const [status, job, detail] = await Promise.all([
        apiJson(`/api/v2/sessions/${sessionId}/status`),
        apiJson(`/api/v2/jobs/${sessionId}`),
        apiJson(`/api/v2/sessions/${sessionId}`),
      ]);
      const diarization = status.diarization || detail.diarization || job.diarization || {};
      const qualityGate = status.quality_gate || detail.quality_gate || job.quality_gate || {};
      el.innerHTML = `
        <div class="grid">
          <div class="section">
            <h3>Lifecycle</h3>
            <div class="kv"><span class="k">State</span><span class="v">${renderBadge(status.state)}</span></div>
            <div class="kv"><span class="k">Backend</span><span class="v">${escHtml(status.backend_outcome || job.backend_outcome || '')}</span></div>
            <div class="kv"><span class="k">Mode</span><span class="v">${escHtml(status.mode || detail.mode || '')}</span></div>
            <div class="kv"><span class="k">Created</span><span class="v">${formatDateTime(status.created_at)}</span></div>
            <div class="kv"><span class="k">Queued</span><span class="v">${formatDateTime(status.queued_at)}</span></div>
            <div class="kv"><span class="k">Started</span><span class="v">${formatDateTime(status.started_at)}</span></div>
            <div class="kv"><span class="k">Finished</span><span class="v">${formatDateTime(status.finished_at)}</span></div>
            <div class="kv"><span class="k">Progress</span><span class="v">${escHtml(status.progress?.processing || 0)}% / ${escHtml(status.progress?.stage || 'pending')}</span></div>
          </div>
          <div class="section">
            <h3>Artifacts</h3>
            <div class="kv"><span class="k">Chunks</span><span class="v">${escHtml(status.chunk_count || 0)}</span></div>
            <div class="kv"><span class="k">Audio</span><span class="v">${formatDurationMs(status.total_audio_ms)}</span></div>
            <div class="kv"><span class="k">Original File</span><span class="v">${escHtml(status.original_filename || 'N/A')}</span></div>
            <div class="kv"><span class="k">Transcript</span><span class="v">${yesNo(status.transcript_present)}</span></div>
            <div class="kv"><span class="k">Partial</span><span class="v">${yesNo(status.partial_transcript_available)}</span></div>
            <div class="kv"><span class="k">Stabilized</span><span class="v">${yesNo(status.stabilized_partial_available)}</span></div>
            <div class="kv"><span class="k">Final</span><span class="v">${yesNo(status.final_transcript_available)}</span></div>
            <div class="kv"><span class="k">Context Spans</span><span class="v">${escHtml(status.context_span_count ?? detail.context_span_count ?? 0)}</span></div>
            <div class="kv"><span class="k">Error</span><span class="v">${escHtml(status.error_message || 'None')}</span></div>
          </div>
        </div>
        <div class="section">
          <h3>Diarization</h3>
          <div class="grid">
            <div>
              <div class="kv"><span class="k">Policy</span><span class="v">${escHtml(diarization.policy || 'auto')}</span></div>
              <div class="kv"><span class="k">Requested</span><span class="v">${yesNo(diarization.requested)}</span></div>
              <div class="kv"><span class="k">Status</span><span class="v">${renderBadge(diarization.status || 'unknown')}</span></div>
            </div>
            <div>
              <div class="kv"><span class="k">Turns</span><span class="v">${escHtml(diarization.turn_count || 0)}</span></div>
              <div class="kv"><span class="k">Speakers</span><span class="v">${escHtml(joinValues(diarization.speakers || [], 'None'))}</span></div>
              <div class="kv"><span class="k">Reason</span><span class="v">${escHtml(diarization.reason || 'N/A')}</span></div>
            </div>
          </div>
        </div>
        <div class="section">
          <h3>Quality Gate</h3>
          <div class="grid">
            <div>
              <div class="kv"><span class="k">Status</span><span class="v">${renderBadge(qualityGate.session_quality_status || 'unknown')}</span></div>
              <div class="kv"><span class="k">Semantic Eligible</span><span class="v">${yesNo(qualityGate.semantic_eligible)}</span></div>
              <div class="kv"><span class="k">Memory Eligible</span><span class="v">${yesNo(qualityGate.memory_update_eligible)}</span></div>
            </div>
            <div class="kv"><span class="k">Reasons</span><span class="v">${renderReasons(qualityGate.reasons)}</span></div>
          </div>
        </div>`;
    } else if (tab === 'transcript') {
      const response = await api(`/api/v2/sessions/${sessionId}/transcript`);
      if (response.status === 202) {
        const pending = await response.json();
        el.innerHTML = `<p>Not ready: ${escHtml(pending.message)}</p>`;
        return;
      }
      const transcript = await response.json();
      el.innerHTML = `
        <div class="transcript-box">${escHtml(transcript.text || '(empty)')}</div>
        <div class="section">
          <h3>Segments (${(transcript.segments || []).length})</h3>
          <table><tr><th>Start</th><th>End</th><th>Speaker</th><th>Text</th></tr>
          ${(transcript.segments || []).slice(0, 50).map((segment) => `<tr><td>${escHtml(segment.start_ms)}</td><td>${escHtml(segment.end_ms)}</td><td>${escHtml(segment.speaker || '')}</td><td>${escHtml(segment.text || '')}</td></tr>`).join('')}
          </table>
        </div>
        <div class="section">
          <div class="kv"><span class="k">Reading Text</span><span class="v">${escHtml(transcript.reading_text || transcript.text || '')}</span></div>
          <div class="kv"><span class="k">Markers</span><span class="v">${escHtml((transcript.markers || []).length)}</span></div>
          <div class="kv"><span class="k">Context Spans</span><span class="v">${escHtml((transcript.context_spans || []).length)}</span></div>
          <div class="kv"><span class="k">Retrieval Entries</span><span class="v">${escHtml(transcript.retrieval_summary?.entry_count || 0)}</span></div>
        </div>
        <div class="section">
          <a href="/api/v2/sessions/${sessionId}/subtitle.srt" target="_blank">Download SRT</a> |
          <a href="/api/v2/sessions/${sessionId}/subtitle.vtt" target="_blank">Download VTT</a>
        </div>`;
    } else if (tab === 'speaker') {
      const response = await api(`/api/v2/sessions/${sessionId}/transcript/speaker`);
      if (response.status === 202) {
        const body = await response.json();
        el.innerHTML = `<p>${escHtml(body.message)}</p>`;
        return;
      }
      if (response.status === 404) {
        const detail = await apiJson(`/api/v2/sessions/${sessionId}`);
        const diarization = detail.diarization || {};
        el.innerHTML = `<p>No speaker transcript available.</p>
          <div class="section">
            <div class="kv"><span class="k">Diarization Status</span><span class="v">${renderBadge(diarization.status || 'unknown')}</span></div>
            <div class="kv"><span class="k">Policy</span><span class="v">${escHtml(diarization.policy || 'auto')}</span></div>
            <div class="kv"><span class="k">Reason</span><span class="v">${escHtml(diarization.reason || 'N/A')}</span></div>
            <div class="kv"><span class="k">Turn Count</span><span class="v">${escHtml(diarization.turn_count || 0)}</span></div>
          </div>`;
        return;
      }
      const speakerTranscript = await response.json();
      el.innerHTML = `<div class="transcript-box">${escHtml(speakerTranscript.speaker_timestamped || '')}</div>`;
    } else if (tab === 'partial') {
      const partial = await apiJson(`/api/v2/sessions/${sessionId}/transcript/partial`);
      el.innerHTML = `
        <div class="kv"><span class="k">Provisional</span><span class="v">${yesNo(partial.provisional)}</span></div>
        <div class="kv"><span class="k">Chunks at Time</span><span class="v">${escHtml(partial.chunk_count_at_time)}</span></div>
        <div class="kv"><span class="k">Generated</span><span class="v">${formatDateTime(partial.generated_at)}</span></div>
        <div class="transcript-box" style="margin-top:8px">${escHtml(partial.text || '(empty)')}</div>`;
    } else if (tab === 'pipeline') {
      const detail = await apiJson(`/api/v2/sessions/${sessionId}`);
      const audit = detail.pipeline_audit;
      if (!audit) {
        el.innerHTML = '<p>No canonical pipeline audit available for this session.</p>';
        return;
      }
      const stages = audit.stages || [];
      el.innerHTML = `
        <div class="grid">
          <div class="section">
            <h3>Run Summary</h3>
            <div class="kv"><span class="k">Run ID</span><span class="v mono">${escHtml(audit.run_id || '')}</span></div>
            <div class="kv"><span class="k">Run Status</span><span class="v">${renderBadge(audit.run_status || 'unknown')}</span></div>
            <div class="kv"><span class="k">Started</span><span class="v">${formatDateTime(audit.started_at)}</span></div>
            <div class="kv"><span class="k">Finished</span><span class="v">${formatDateTime(audit.finished_at)}</span></div>
            <div class="kv"><span class="k">Error</span><span class="v">${escHtml(audit.error || 'None')}</span></div>
          </div>
          <div class="section">
            <h3>Stage Metrics</h3>
            <div class="kv"><span class="k">Decode Windows</span><span class="v">${escHtml(audit.decode_lattice?.scheduled_windows ?? audit.decode_lattice?.total_windows ?? 'N/A')}</span></div>
            <div class="kv"><span class="k">ASR Candidates</span><span class="v">${escHtml(audit.asr?.total_candidates ?? 'N/A')}</span></div>
            <div class="kv"><span class="k">Stripes</span><span class="v">${escHtml(audit.reconciliation?.stripe_count ?? 'N/A')}</span></div>
            <div class="kv"><span class="k">LLM Resolved</span><span class="v">${escHtml(audit.reconciliation?.llm_resolved ?? 'N/A')}</span></div>
            <div class="kv"><span class="k">Fallback Resolved</span><span class="v">${escHtml(audit.reconciliation?.fallback_resolved ?? 'N/A')}</span></div>
          </div>
        </div>
        <div class="section">
          <h3>Stages</h3>
          <table><tr><th>Stage</th><th>Status</th><th>Model</th><th>Routing</th><th>Error</th></tr>
          ${stages.map((stage) => `<tr>
            <td>${escHtml(stage.name)}</td>
            <td>${renderBadge(stage.status || 'pending')}</td>
            <td>${escHtml(stage.actual_model || stage.selected_model || '')}</td>
            <td>${escHtml(stage.routing_reason || '')}</td>
            <td>${escHtml(stage.error || '')}</td>
          </tr>`).join('')}
          </table>
        </div>
        <div class="section">
          <h3>Raw Session Detail</h3>
          <div class="pre-box mono">${escHtml(JSON.stringify(detail, null, 2))}</div>
        </div>`;
    }
  } catch (error) {
    el.innerHTML = `<p style="color:var(--red)">Failed to load ${escHtml(tab)} tab: ${escHtml(error.message)}</p>`;
  }
}

function onUploadSelectionChanged() {
  const files = Array.from(document.getElementById('upload-file').files || []);
  const summary = document.getElementById('upload-selection-summary');
  if (!files.length) {
    summary.textContent = 'No files selected.';
    return;
  }
  const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
  summary.innerHTML = `
    <div><strong>${files.length}</strong> file(s) selected</div>
    <div class="small muted">${files.map((file) => escHtml(file.name)).join(', ')}</div>
    <div class="small muted">Total size: ${formatBytes(totalBytes)}</div>`;
}

function clearUploadSelection() {
  document.getElementById('upload-file').value = '';
  onUploadSelectionChanged();
}

function renderUploadQueue() {
  const status = document.getElementById('upload-status');
  if (!uploadQueue.length) {
    status.innerHTML = '<p class="muted">No queued uploads yet.</p>';
    return;
  }

  const counts = {
    waiting: uploadQueue.filter((item) => item.localState === 'waiting').length,
    uploading: uploadQueue.filter((item) => item.localState === 'uploading').length,
    active: uploadQueue.filter((item) => ['queued', 'running'].includes(item.localState)).length,
    done: uploadQueue.filter((item) => item.localState === 'done').length,
    error: uploadQueue.filter((item) => item.localState === 'error').length,
  };

  let html = `
    <div class="queue-summary">
      <div><strong>Queue</strong></div>
      <div class="small muted">Waiting: ${counts.waiting} | Uploading: ${counts.uploading} | Processing: ${counts.active} | Done: ${counts.done} | Error: ${counts.error}</div>
    </div>
    <table class="queue-table">
      <tr><th>File</th><th>Status</th><th>Progress</th><th>Session</th><th></th></tr>`;
  uploadQueue.forEach((item) => {
    const progressText = item.localState === 'uploading'
      ? 'Uploading...'
      : item.localState === 'waiting'
        ? 'Waiting to upload'
        : `${item.progress || 0}%${item.stage ? ` / ${escHtml(item.stage)}` : ''}`;
    const details = [
      item.chunkCount ? `${item.chunkCount} transport chunk(s)` : null,
      item.error ? `Error: ${escHtml(item.error)}` : null,
      item.backendOutcome ? `Backend: ${escHtml(item.backendOutcome)}` : null,
      item.finishedAt ? `Finished: ${formatDateTime(item.finishedAt)}` : null,
    ].filter(Boolean);
    html += `<tr>
      <td>
        <div>${escHtml(item.name)}</div>
        <div class="small muted">${formatBytes(item.size)}</div>
        ${details.length ? `<div class="small muted">${details.join(' | ')}</div>` : ''}
      </td>
      <td>${renderBadge(item.localState, item.serverState || item.localState)}</td>
      <td>${progressText}</td>
      <td>${item.sessionId ? `<span class="mono">${escHtml(item.sessionId.slice(0, 8))}...</span>` : '-'}</td>
      <td>${item.sessionId ? `<a onclick="openSession('${item.sessionId}')">Open</a>` : ''}</td>
    </tr>`;
  });
  html += '</table>';
  status.innerHTML = html;
}

function ensureUploadQueuePolling() {
  const hasActive = uploadQueue.some((item) => item.sessionId && !['done', 'error'].includes(item.localState));
  if (hasActive && !uploadQueuePollHandle) {
    uploadQueuePollHandle = setInterval(refreshUploadQueue, 2000);
  } else if (!hasActive && uploadQueuePollHandle) {
    clearInterval(uploadQueuePollHandle);
    uploadQueuePollHandle = null;
  }
}

async function refreshUploadQueue() {
  const activeItems = uploadQueue.filter((item) => item.sessionId && !['done', 'error'].includes(item.localState));
  if (!activeItems.length) {
    ensureUploadQueuePolling();
    renderUploadQueue();
    return;
  }
  await Promise.all(activeItems.map(async (item) => {
    try {
      const job = await apiJson(`/api/v2/jobs/${item.sessionId}`);
      item.serverState = job.state || item.serverState;
      item.localState = (job.state === 'done' || job.state === 'error') ? job.state : (job.state || item.localState);
      item.progress = job.progress?.processing || 0;
      item.stage = job.progress?.stage || item.stage;
      item.backendOutcome = job.backend_outcome;
      item.queuedAt = job.queued_at;
      item.startedAt = job.started_at;
      item.finishedAt = job.finished_at;
      if (job.error) {
        item.error = job.error.error_message || job.error.error || item.error;
      }
    } catch (error) {
      item.error = error.message;
    }
  }));
  ensureUploadQueuePolling();
  renderUploadQueue();
}

async function processUploadQueue() {
  if (uploadQueueProcessing) return;
  uploadQueueProcessing = true;
  try {
    while (true) {
      const next = uploadQueue.find((item) => item.localState === 'waiting' && item.file);
      if (!next) break;
      next.localState = 'uploading';
      next.serverState = 'uploading';
      next.progress = 0;
      next.stage = 'upload';
      renderUploadQueue();

      const formData = new FormData();
      formData.append('file', next.file);
      formData.append('language', document.getElementById('upload-lang').value);
      formData.append('model_size', document.getElementById('upload-model').value);
      const diarizationPolicy = document.getElementById('upload-diarization').value;
      formData.append('diarization', String(diarizationPolicy === 'forced'));
      formData.append('diarization_policy', diarizationPolicy);

      try {
        const upload = await apiJson('/api/file-upload', { method: 'POST', body: formData });
        next.sessionId = upload.session_id;
        next.chunkCount = upload.transport_chunk_count || 1;
        next.localState = 'queued';
        next.serverState = 'queued';
        next.stage = 'queued';
        next.file = null;
      } catch (error) {
        next.localState = 'error';
        next.serverState = 'error';
        next.error = error.message;
        next.file = null;
      }
      renderUploadQueue();
      ensureUploadQueuePolling();
    }
  } finally {
    uploadQueueProcessing = false;
    renderUploadQueue();
  }
}

async function startUpload() {
  const files = Array.from(document.getElementById('upload-file').files || []);
  if (!files.length) {
    if (uploadQueue.some((item) => item.localState === 'waiting' && item.file)) {
      processUploadQueue();
      return;
    }
    document.getElementById('upload-status').innerHTML = '<p>Select one or more files first.</p>';
    return;
  }

  files.forEach((file) => {
    uploadQueue.push({
      id: `upload_${++uploadQueueCounter}`,
      name: file.name,
      size: file.size || 0,
      file,
      localState: 'waiting',
      serverState: 'waiting',
      progress: 0,
      stage: 'waiting',
      sessionId: null,
      chunkCount: null,
      error: null,
      queuedAt: null,
      startedAt: null,
      finishedAt: null,
      backendOutcome: null,
    });
  });
  clearUploadSelection();
  renderUploadQueue();
  processUploadQueue();
}

function loadModels() {
  apiJson('/api/v2/models').then((payload) => {
    let html = '<table><tr><th>Model ID</th><th>Provider</th><th>Status</th><th>Installed</th><th>Languages</th><th>Notes</th></tr>';
    (payload.models || []).forEach((model) => {
      html += `<tr><td class="mono">${escHtml(model.model_id)}</td><td>${escHtml(model.provider || '')}</td>
        <td>${renderBadge(model.provider_status || 'unknown')}</td>
        <td>${yesNo(model.installed)}</td>
        <td>${escHtml(joinValues(model.languages || [], 'All'))}</td>
        <td style="max-width:200px;font-size:12px">${escHtml(model.notes || '')}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('models-list').innerHTML = html;
  }).catch((error) => {
    document.getElementById('models-list').innerHTML = `<p style="color:var(--red)">Failed to load models: ${escHtml(error.message)}</p>`;
  });
}

function loadDiagnostics() {
  Promise.all([
    apiJson('/api/diagnostics'),
    apiJson('/api/v2/system/gpu'),
  ]).then(([diagnostics, gpu]) => {
    document.getElementById('diag-info').innerHTML = `
      <div class="diag-grid">
        <div class="section">
          <h3>Redis</h3>
          <div class="kv"><span class="k">Connected</span><span class="v">${yesNo(diagnostics.redis?.ok)}</span></div>
          <div class="kv"><span class="k">Main Queue</span><span class="v">${escHtml(diagnostics.redis?.queue || '?')} (${escHtml(diagnostics.redis?.queue_len || 0)})</span></div>
          <div class="kv"><span class="k">Partial Queue</span><span class="v">${escHtml(diagnostics.redis?.partial_queue || '?')} (${escHtml(diagnostics.redis?.partial_queue_len || 0)})</span></div>
          <div class="kv"><span class="k">Error</span><span class="v">${escHtml(diagnostics.redis?.error || 'None')}</span></div>
        </div>
        <div class="section">
          <h3>GPU / Worker</h3>
          <div class="kv"><span class="k">Available</span><span class="v">${yesNo(gpu.gpu_available)}</span></div>
          <div class="kv"><span class="k">Reason</span><span class="v">${escHtml(gpu.gpu_reason || 'N/A')}</span></div>
          <div class="kv"><span class="k">Device</span><span class="v">${escHtml(gpu.selected_device || 'cpu')} / ${escHtml(gpu.selected_compute_type || 'int8')}</span></div>
          <div class="kv"><span class="k">GPU Name</span><span class="v">${escHtml(gpu.gpu_name || 'N/A')}</span></div>
          <div class="kv"><span class="k">Memory</span><span class="v">${escHtml(gpu.memory_used_mb ?? '?')}/${escHtml(gpu.memory_total_mb ?? '?')} MB</span></div>
          <div class="kv"><span class="k">Worker Started</span><span class="v">${formatDateTime(gpu.worker_started_at)}</span></div>
          <div class="kv"><span class="k">Queue Depth</span><span class="v">${escHtml(gpu.queue_depth || 0)} main / ${escHtml(gpu.partial_queue_depth || 0)} partial</span></div>
        </div>
      </div>
      <div class="diag-grid section">
        <div class="section">
          <h3>Server / TLS</h3>
          <div class="kv"><span class="k">Host</span><span class="v">${escHtml(diagnostics.server?.host || '')}</span></div>
          <div class="kv"><span class="k">Port</span><span class="v">${escHtml(diagnostics.server?.port || '')}</span></div>
          <div class="kv"><span class="k">TLS Enabled</span><span class="v">${yesNo(diagnostics.tls?.enabled)}</span></div>
          <div class="kv"><span class="k">Cert Exists</span><span class="v">${yesNo(diagnostics.tls?.cert_exists)}</span></div>
          <div class="kv"><span class="k">Key Exists</span><span class="v">${yesNo(diagnostics.tls?.key_exists)}</span></div>
        </div>
        <div class="section">
          <h3>Storage</h3>
          <div class="kv"><span class="k">Sessions Dir</span><span class="v mono">${escHtml(diagnostics.sessions_dir || '?')}</span></div>
          <div class="kv"><span class="k">Models Dir</span><span class="v mono">${escHtml(diagnostics.models_dir || '?')}</span></div>
          <div class="kv"><span class="k">Models Dir Exists</span><span class="v">${yesNo(diagnostics.models_dir_exists)}</span></div>
          <div class="kv"><span class="k">Timestamp</span><span class="v">${formatDateTime(diagnostics.timestamp)}</span></div>
        </div>
      </div>
      <div class="section">
        <h3>Backend Status</h3>
        <div class="pre-box mono">${escHtml(JSON.stringify(gpu.backend_status || diagnostics.gpu?.backend_status || {}, null, 2))}</div>
      </div>`;
  }).catch((error) => {
    document.getElementById('diag-info').innerHTML = `<p style="color:var(--red)">Failed to load diagnostics: ${escHtml(error.message)}</p>`;
  });
}

async function deleteSession(sessionId) {
  if (!confirm('Delete session ' + sessionId + '?')) return;
  await api(`/api/v2/sessions/${sessionId}`, { method: 'DELETE' });
  uploadQueue = uploadQueue.filter((item) => item.sessionId !== sessionId);
  ensureUploadQueuePolling();
  if (currentSession === sessionId) {
    showPage('sessions');
  } else {
    renderUploadQueue();
    loadSessions();
  }
}

async function retrySession(sessionId) {
  try {
    await apiJson(`/api/v2/sessions/${sessionId}/retry`, { method: 'POST' });
    openSession(sessionId);
  } catch (error) {
    alert('Retry failed: ' + error.message);
  }
}

window.submitToken = submitToken;
window.logout = logout;
window.showPage = showPage;
window.openSession = openSession;
window.showSessionTab = showSessionTab;
window.startUpload = startUpload;
window.clearUploadSelection = clearUploadSelection;
window.onUploadSelectionChanged = onUploadSelectionChanged;
window.deleteSession = deleteSession;
window.retrySession = retrySession;

if (TOKEN) {
  apiJson('/api/v2/health').then(() => {
    document.getElementById('auth-gate').classList.add('hidden');
    showPage('dashboard');
  }).catch(() => showAuthGate());
} else {
  showAuthGate();
}
