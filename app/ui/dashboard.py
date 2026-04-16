"""
Browser UI - Single-page application served at /

Features:
  - Token gate / auth input (localStorage, never in URL)
  - Sessions list and grouped view
  - Session detail page with pipeline/provenance/debug
  - Manual file upload/test
  - Live processing status
  - Partial, final, and speaker transcript views
  - SRT/VTT download links
  - Model registry viewer
  - Diagnostics page
"""


def get_ui_html() -> str:
    return _UI_HTML


_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LiveVoiceTranscriptor</title>
<style>
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#c9d1d9;--text-dim:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;--radius:6px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.container{max-width:1100px;margin:0 auto;padding:16px}
header{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 20px;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:18px;font-weight:600}
header .nav{display:flex;gap:16px}
header .nav a{color:var(--text-dim);font-size:14px;cursor:pointer}
header .nav a.active,header .nav a:hover{color:var(--accent)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:16px}
.card h2{font-size:16px;margin-bottom:12px;color:var(--accent)}
.card h3{font-size:14px;margin-bottom:8px}
input,select,textarea{background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:var(--radius);font-size:14px;width:100%}
input:focus,select:focus{outline:none;border-color:var(--accent)}
button{background:var(--accent);color:#fff;border:none;padding:8px 16px;border-radius:var(--radius);cursor:pointer;font-size:14px}
button:hover{opacity:0.9}
button.danger{background:var(--red)}
button.secondary{background:var(--border);color:var(--text)}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:500}
.badge.done{background:#1a3a1a;color:var(--green)}
.badge.running{background:#1a2a3a;color:var(--accent)}
.badge.error{background:#3a1a1a;color:var(--red)}
.badge.queued{background:#2a2a1a;color:var(--yellow)}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--border);font-size:13px}
th{color:var(--text-dim);font-weight:500}
.mono{font-family:'Cascadia Code','Fira Code',monospace;font-size:12px}
.transcript-box{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:12px;max-height:400px;overflow-y:auto;white-space:pre-wrap;font-size:13px;line-height:1.8}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
.status-bar{display:flex;gap:8px;align-items:center;margin-bottom:12px}
.progress-bar{height:6px;background:var(--border);border-radius:3px;flex:1}
.progress-fill{height:100%;background:var(--accent);border-radius:3px;transition:width 0.3s}
#auth-gate{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:80vh;gap:16px}
#auth-gate input{max-width:400px}
.hidden{display:none!important}
.tab-bar{display:flex;gap:8px;margin-bottom:16px;border-bottom:1px solid var(--border);padding-bottom:8px}
.tab-bar button{background:transparent;color:var(--text-dim);border:none;padding:6px 12px;border-radius:var(--radius) var(--radius) 0 0;cursor:pointer}
.tab-bar button.active{color:var(--accent);border-bottom:2px solid var(--accent)}
.upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:40px;text-align:center;cursor:pointer}
.upload-zone:hover{border-color:var(--accent)}
.stack{display:flex;flex-direction:column;gap:12px}
.muted{color:var(--text-dim)}
.small{font-size:12px}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border:1px solid var(--border);border-radius:999px;font-size:12px}
.queue-summary{padding:10px 12px;border:1px solid var(--border);border-radius:var(--radius);background:rgba(88,166,255,0.06)}
.queue-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.queue-actions button{flex:1;min-width:180px}
.queue-table{width:100%;border-collapse:collapse;margin-top:8px}
.queue-table th,.queue-table td{padding:8px 10px;vertical-align:top}
.queue-table td:last-child,.queue-table th:last-child{text-align:right}
.diag-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.section{margin-top:16px}
.section:first-child{margin-top:0}
.pre-box{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:12px;overflow:auto;max-height:360px;white-space:pre-wrap}
@media(max-width:768px){.diag-grid{grid-template-columns:1fr}}
.kv{display:flex;gap:8px;margin-bottom:4px}
.kv .k{color:var(--text-dim);min-width:160px;font-size:13px}
.kv .v{font-size:13px}
</style>
</head>
<body>
<header>
  <h1>LiveVoiceTranscriptor</h1>
  <div class="nav">
    <a onclick="showPage('dashboard')" id="nav-dashboard">Dashboard</a>
    <a onclick="showPage('sessions')" id="nav-sessions">Sessions</a>
    <a onclick="showPage('upload')" id="nav-upload">Upload</a>
    <a onclick="showPage('models')" id="nav-models">Models</a>
    <a onclick="showPage('diagnostics')" id="nav-diagnostics">Diagnostics</a>
    <a onclick="logout()" style="color:var(--red)">Logout</a>
  </div>
</header>

<div class="container">
  <!-- Auth Gate -->
  <div id="auth-gate">
    <h2>Authentication Required</h2>
    <input type="password" id="token-input" placeholder="Enter API token..." onkeydown="if(event.key==='Enter')submitToken()">
    <button onclick="submitToken()">Connect</button>
    <div id="auth-error" style="color:var(--red)"></div>
  </div>

  <!-- Dashboard -->
  <div id="page-dashboard" class="hidden">
    <div class="grid">
      <div class="card">
        <h2>System Health</h2>
        <div id="health-info">Loading...</div>
      </div>
      <div class="card">
        <h2>GPU Status</h2>
        <div id="gpu-info">Loading...</div>
      </div>
    </div>
    <div class="card">
      <h2>Recent Sessions</h2>
      <div id="recent-sessions">Loading...</div>
    </div>
  </div>

  <!-- Sessions -->
  <div id="page-sessions" class="hidden">
    <div class="card">
      <h2>All Sessions</h2>
      <div id="sessions-grouped">Loading...</div>
    </div>
  </div>

  <!-- Session Detail -->
  <div id="page-session-detail" class="hidden">
    <div class="card" id="session-header"></div>
    <div class="tab-bar">
      <button class="active" data-session-tab="status" onclick="showSessionTab('status', this)">Status</button>
      <button data-session-tab="transcript" onclick="showSessionTab('transcript', this)">Transcript</button>
      <button data-session-tab="speaker" onclick="showSessionTab('speaker', this)">Speaker</button>
      <button data-session-tab="partial" onclick="showSessionTab('partial', this)">Partial</button>
      <button data-session-tab="pipeline" onclick="showSessionTab('pipeline', this)">Pipeline</button>
    </div>
    <div id="session-tab-content" class="card"></div>
  </div>

  <!-- Upload -->
  <div id="page-upload" class="hidden">
    <div class="card">
      <h2>Upload Audio Files</h2>
      <p style="color:var(--text-dim);margin-bottom:16px">Choose one or more audio files. The UI will upload them one by one and place each session into the server queue.</p>
      <div class="grid">
        <div class="stack">
          <input type="file" id="upload-file" accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac,.aac,.webm" multiple onchange="onUploadSelectionChanged()">
          <div style="margin-top:12px">
            <label style="font-size:13px;color:var(--text-dim)">Model</label>
            <select id="upload-model" style="margin-top:4px">
              <option value="auto">Auto (Canonical Pipeline)</option>
              <option value="small">faster-whisper:small</option>
              <option value="medium">faster-whisper:medium</option>
              <option value="large-v3">faster-whisper:large-v3</option>
            </select>
          </div>
          <div style="margin-top:12px">
            <label style="font-size:13px;color:var(--text-dim)">Language</label>
            <select id="upload-lang" style="margin-top:4px">
              <option value="auto">Auto-detect</option>
              <option value="en">English</option>
              <option value="ru">Russian</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="es">Spanish</option>
            </select>
          </div>
          <div style="margin-top:12px">
            <label style="font-size:13px;color:var(--text-dim)">Diarization</label>
            <select id="upload-diarization" style="margin-top:4px">
              <option value="auto">Auto</option>
              <option value="off">Off</option>
              <option value="forced">Forced</option>
            </select>
          </div>
          <div id="upload-selection-summary" class="queue-summary small">No files selected.</div>
          <div class="queue-actions">
            <button onclick="startUpload()">Queue Selected Files</button>
            <button class="secondary" onclick="clearUploadSelection()">Clear Selection</button>
          </div>
        </div>
        <div>
          <div id="upload-status" style="font-size:13px"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Models -->
  <div id="page-models" class="hidden">
    <div class="card">
      <h2>Model Registry</h2>
      <div id="models-list">Loading...</div>
    </div>
  </div>

  <!-- Diagnostics -->
  <div id="page-diagnostics" class="hidden">
    <div class="card">
      <h2>System Diagnostics</h2>
      <div id="diag-info">Loading...</div>
    </div>
  </div>
</div>

<script>
window.__LVT_USE_EXTERNAL_DASHBOARD__=true;
const LS_KEY='lvt_token';
let TOKEN=localStorage.getItem(LS_KEY)||'';
let currentSession=null;

function api(path,opts={}){
  const h={'Authorization':'Bearer '+TOKEN,...(opts.headers||{})};
  return fetch(path,{...opts,headers:h}).then(r=>{
    if(r.status===401||r.status===403){showAuthGate();throw new Error('auth');}
    return r;
  });
}
function apiJson(path,opts){return api(path,opts).then(r=>r.json());}

function submitToken(){
  TOKEN=document.getElementById('token-input').value.trim();
  if(!TOKEN){document.getElementById('auth-error').textContent='Token required';return;}
  apiJson('/api/v2/health').then(d=>{
    localStorage.setItem(LS_KEY,TOKEN);
    document.getElementById('auth-gate').classList.add('hidden');
    showPage('dashboard');
  }).catch(e=>{
    document.getElementById('auth-error').textContent='Invalid token or server unreachable';
  });
}

function showAuthGate(){
  document.querySelectorAll('[id^="page-"]').forEach(p=>p.classList.add('hidden'));
  document.getElementById('auth-gate').classList.remove('hidden');
}

function logout(){TOKEN='';localStorage.removeItem(LS_KEY);showAuthGate();}

function showPage(page){
  document.querySelectorAll('[id^="page-"]').forEach(p=>p.classList.add('hidden'));
  document.querySelectorAll('.nav a').forEach(a=>a.classList.remove('active'));
  const el=document.getElementById('page-'+page);
  if(el)el.classList.remove('hidden');
  const nav=document.getElementById('nav-'+page);
  if(nav)nav.classList.add('active');
  if(page==='dashboard')loadDashboard();
  else if(page==='sessions')loadSessions();
  else if(page==='models')loadModels();
  else if(page==='diagnostics')loadDiagnostics();
}

function loadDashboard(){
  apiJson('/api/v2/health').then(d=>{
    document.getElementById('health-info').innerHTML=`
      <div class="kv"><span class="k">Version</span><span class="v">${d.version}</span></div>
      <div class="kv"><span class="k">GPU</span><span class="v">${d.gpu_available?'Available':'Not available'} (${d.gpu_reason||''})</span></div>
      <div class="kv"><span class="k">Device</span><span class="v">${d.selected_device}</span></div>
      <div class="kv"><span class="k">Compute</span><span class="v">${d.selected_compute_type}</span></div>`;
  }).catch(()=>{document.getElementById('health-info').textContent='Failed to load';});

  apiJson('/api/v2/system/gpu').then(d=>{
    document.getElementById('gpu-info').innerHTML=`
      <div class="kv"><span class="k">GPU Name</span><span class="v">${d.gpu_name||'N/A'}</span></div>
      <div class="kv"><span class="k">Memory</span><span class="v">${d.memory_used_mb||'?'}/${d.memory_total_mb||'?'} MB</span></div>
      <div class="kv"><span class="k">Queue</span><span class="v">${d.queue_depth} jobs</span></div>
      <div class="kv"><span class="k">Active</span><span class="v">${d.active_job_count} jobs</span></div>`;
  }).catch(()=>{document.getElementById('gpu-info').textContent='Failed to load';});

  apiJson('/api/sessions').then(d=>{
    const s=d.sessions||[];
    if(!s.length){document.getElementById('recent-sessions').textContent='No sessions yet.';return;}
    let html='<table><tr><th>Session</th><th>Status</th><th>Mode</th><th>Created</th><th></th></tr>';
    s.slice(0,10).forEach(sess=>{
      html+=`<tr><td class="mono">${sess.session_id.slice(0,8)}...</td>
        <td><span class="badge ${sess.status}">${sess.status}</span></td>
        <td>${sess.mode||''}</td>
        <td>${sess.created_at?new Date(sess.created_at).toLocaleString():''}</td>
        <td><a onclick="openSession('${sess.session_id}')">View</a></td></tr>`;
    });
    html+='</table>';
    document.getElementById('recent-sessions').innerHTML=html;
  }).catch(()=>{document.getElementById('recent-sessions').textContent='Failed';});
}

function loadSessions(){
  apiJson('/api/v2/sessions/grouped').then(d=>{
    let html='';
    (d.groups||[]).forEach(g=>{
      html+=`<h3 style="margin:16px 0 8px">${g.label} (${g.sessions.length})</h3>`;
      html+='<table><tr><th>Session</th><th>State</th><th>Mode</th><th>Chunks</th><th>Transcript</th><th></th></tr>';
      g.sessions.forEach(s=>{
        html+=`<tr><td class="mono">${s.session_id.slice(0,8)}...</td>
          <td><span class="badge ${s.state}">${s.state}</span></td>
          <td>${s.mode||''}</td><td>${s.chunk_count||0}</td>
          <td>${s.transcript_present?'Yes':'No'}</td>
          <td><a onclick="openSession('${s.session_id}')">View</a>
              <a onclick="deleteSession('${s.session_id}')" style="color:var(--red);margin-left:8px">Del</a></td></tr>`;
      });
      html+='</table>';
    });
    if(!html)html='<p>No sessions.</p>';
    html+=`<p style="margin-top:12px;color:var(--text-dim)">Total: ${d.total_sessions||0}</p>`;
    document.getElementById('sessions-grouped').innerHTML=html;
  });
}

function openSession(sid){
  currentSession=sid;
  document.querySelectorAll('[id^="page-"]').forEach(p=>p.classList.add('hidden'));
  document.getElementById('page-session-detail').classList.remove('hidden');
  loadSessionDetail(sid);
}

function loadSessionDetail(sid){
  apiJson(`/api/v2/jobs/${sid}`).then(d=>{
    document.getElementById('session-header').innerHTML=`
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div><h2 class="mono">${sid}</h2>
          <span class="badge ${d.state}">${d.state}</span>
          <span style="color:var(--text-dim);margin-left:8px">${d.backend_outcome||''}</span></div>
        <div>
          <button class="secondary" onclick="showPage('sessions')">Back</button>
          ${d.state==='error'?`<button onclick="retrySession('${sid}')" style="margin-left:8px">Retry</button>`:''}
          <button class="danger" onclick="deleteSession('${sid}')" style="margin-left:8px">Delete</button>
        </div>
      </div>
      <div class="status-bar" style="margin-top:12px">
        <div class="progress-bar"><div class="progress-fill" style="width:${d.progress?.processing||0}%"></div></div>
        <span style="font-size:12px">${d.progress?.processing||0}% - ${d.progress?.stage||'pending'}</span>
      </div>
      <div class="grid" style="margin-top:8px">
        <div>
          <div class="kv"><span class="k">Queued</span><span class="v">${d.queued_at||'N/A'}</span></div>
          <div class="kv"><span class="k">Started</span><span class="v">${d.started_at||'N/A'}</span></div>
          <div class="kv"><span class="k">Finished</span><span class="v">${d.finished_at||'N/A'}</span></div>
        </div>
        <div>
          <div class="kv"><span class="k">Model</span><span class="v">${d.meta?.model_size||'auto'}</span></div>
          <div class="kv"><span class="k">Language</span><span class="v">${d.meta?.language_requested||'auto'}</span></div>
          <div class="kv"><span class="k">Diarization</span><span class="v">${d.meta?.diarization_enabled?'Yes':'No'}</span></div>
        </div>
      </div>`;
    showSessionTab('status');
  }).catch(()=>{document.getElementById('session-header').innerHTML='<p>Failed to load session.</p>';});
}

function showSessionTab(tab){
  document.querySelectorAll('.tab-bar button').forEach(b=>b.classList.remove('active'));
  event?.target?.classList?.add('active');
  const sid=currentSession;if(!sid)return;
  const el=document.getElementById('session-tab-content');

  if(tab==='status'){
    apiJson(`/api/v2/sessions/${sid}/status`).then(d=>{
      el.innerHTML=`
        <div class="grid">
          <div>
            <div class="kv"><span class="k">State</span><span class="v">${d.state}</span></div>
            <div class="kv"><span class="k">Chunks</span><span class="v">${d.chunk_count}</span></div>
            <div class="kv"><span class="k">Audio</span><span class="v">${d.total_audio_ms}ms</span></div>
            <div class="kv"><span class="k">Mode</span><span class="v">${d.mode}</span></div>
          </div>
          <div>
            <div class="kv"><span class="k">Transcript</span><span class="v">${d.transcript_present?'Yes':'No'}</span></div>
            <div class="kv"><span class="k">Partial</span><span class="v">${d.partial_transcript_available?'Yes':'No'}</span></div>
            <div class="kv"><span class="k">Error</span><span class="v">${d.error_message||'None'}</span></div>
          </div>
        </div>`;
    });
  } else if(tab==='transcript'){
    api(`/api/v2/sessions/${sid}/transcript`).then(r=>{
      if(r.status===202)return r.json().then(d=>{el.innerHTML=`<p>Not ready: ${d.message}</p>`;});
      return r.json().then(d=>{
        el.innerHTML=`
          <div class="transcript-box">${escHtml(d.text||'(empty)')}</div>
          <div style="margin-top:12px">
            <h3>Segments (${(d.segments||[]).length})</h3>
            <table><tr><th>Start</th><th>End</th><th>Speaker</th><th>Text</th></tr>
            ${(d.segments||[]).slice(0,50).map(s=>`<tr><td>${s.start_ms}</td><td>${s.end_ms}</td><td>${s.speaker||''}</td><td>${escHtml(s.text||'')}</td></tr>`).join('')}
            </table>
          </div>
          <div style="margin-top:12px">
            <a href="/api/v2/sessions/${sid}/subtitle.srt" target="_blank">Download SRT</a> |
            <a href="/api/v2/sessions/${sid}/subtitle.vtt" target="_blank">Download VTT</a>
          </div>`;
      });
    });
  } else if(tab==='speaker'){
    api(`/api/v2/sessions/${sid}/transcript/speaker`).then(r=>{
      if(r.status===202)return r.json().then(d=>{el.innerHTML=`<p>${d.message}</p>`;});
      if(r.status===404)return el.innerHTML='<p>No speaker transcript available.</p>';
      return r.json().then(d=>{el.innerHTML=`<div class="transcript-box">${escHtml(d.speaker_timestamped||'')}</div>`;});
    });
  } else if(tab==='partial'){
    apiJson(`/api/v2/sessions/${sid}/transcript/partial`).then(d=>{
      el.innerHTML=`
        <div class="kv"><span class="k">Provisional</span><span class="v">${d.provisional}</span></div>
        <div class="kv"><span class="k">Chunks at time</span><span class="v">${d.chunk_count_at_time}</span></div>
        <div class="transcript-box" style="margin-top:8px">${escHtml(d.text||'(empty)')}</div>`;
    }).catch(()=>{el.innerHTML='<p>No partial transcript available.</p>';});
  } else if(tab==='pipeline'){
    apiJson(`/api/v2/sessions/${sid}`).then(d=>{
      el.innerHTML=`<h3>Session Details</h3><pre class="mono" style="background:var(--bg);padding:12px;border-radius:var(--radius);overflow:auto;max-height:400px">${escHtml(JSON.stringify(d,null,2))}</pre>`;
    });
  }
}

async function startUpload(){
  const fileInput=document.getElementById('upload-file');
  const model=document.getElementById('upload-model').value;
  const lang=document.getElementById('upload-lang').value;
  const status=document.getElementById('upload-status');
  if(!fileInput.files.length){status.textContent='Select a file first.';return;}

  const file=fileInput.files[0];
  status.innerHTML='<p>Uploading file...</p>';

  try{
    const fd=new FormData();
    fd.append('file',file);
    fd.append('language',lang);
    fd.append('model_size',model==='auto'?'auto':model);
    fd.append('diarization','false');
    const upload=await apiJson('/api/file-upload',{method:'POST',body:fd});
    const sid=upload.session_id;
    const chunkCount=upload.transport_chunk_count||1;
    status.innerHTML+=`<p>Session: ${sid.slice(0,8)}...</p><p>Server split into ${chunkCount} transport chunk(s).</p><p>Processing... <a onclick="openSession('${sid}')">View session</a></p>`;

    // Poll
    let tries=0;
    const poll=setInterval(async()=>{
      tries++;
      const j=await apiJson(`/api/v2/jobs/${sid}`);
      status.innerHTML=`<p>Session: <a onclick="openSession('${sid}')">${sid.slice(0,8)}...</a></p>
        <p>State: <span class="badge ${j.state}">${j.state}</span> ${j.progress?.processing||0}%</p>
        <p>Stage: ${j.progress?.stage||'pending'}</p>`;
      if(j.state==='done'||j.state==='error'||tries>120){
        clearInterval(poll);
        if(j.state==='done')status.innerHTML+=`<p style="color:var(--green)">Done! <a onclick="openSession('${sid}')">View transcript</a></p>`;
        else if(j.state==='error')status.innerHTML+=`<p style="color:var(--red)">Failed.</p>`;
      }
    },2000);
  }catch(e){status.innerHTML=`<p style="color:var(--red)">Error: ${e.message}</p>`;}
}

function loadModels(){
  apiJson('/api/v2/models').then(d=>{
    let html='<table><tr><th>Model ID</th><th>Provider</th><th>Status</th><th>Installed</th><th>Languages</th><th>Notes</th></tr>';
    (d.models||[]).forEach(m=>{
      const statusClass=m.provider_status==='stable'?'done':m.provider_status==='experimental'?'running':'error';
      html+=`<tr><td class="mono">${m.model_id}</td><td>${m.provider}</td>
        <td><span class="badge ${statusClass}">${m.provider_status}</span></td>
        <td>${m.installed?'Yes':'No'}</td>
        <td>${m.languages?m.languages.join(', '):'All'}</td>
        <td style="max-width:200px;font-size:12px">${m.notes||''}</td></tr>`;
    });
    html+='</table>';
    document.getElementById('models-list').innerHTML=html;
  });
}

function loadDiagnostics(){
  apiJson('/api/diagnostics').then(d=>{
    document.getElementById('diag-info').innerHTML=`
      <h3>Redis</h3>
      <div class="kv"><span class="k">Connected</span><span class="v">${d.redis?.ok?'Yes':'No'}</span></div>
      <div class="kv"><span class="k">Queue Length</span><span class="v">${d.redis?.queue_len||0}</span></div>
      ${d.redis?.error?`<div class="kv"><span class="k">Error</span><span class="v" style="color:var(--red)">${d.redis.error}</span></div>`:''}
      <h3 style="margin-top:12px">GPU</h3>
      <div class="kv"><span class="k">Available</span><span class="v">${d.gpu?.available?'Yes':'No'}${d.gpu?.reason?` (${d.gpu.reason})`:''}</span></div>
      ${d.gpu?.name?`<div class="kv"><span class="k">Name</span><span class="v">${d.gpu.name}</span></div>`:''}
      ${d.gpu?.selected_device?`<div class="kv"><span class="k">Worker Device</span><span class="v">${d.gpu.selected_device}${d.gpu.selected_compute_type?` / ${d.gpu.selected_compute_type}`:''}</span></div>`:''}
      <h3 style="margin-top:12px">Storage</h3>
      <div class="kv"><span class="k">Sessions Dir</span><span class="v mono">${d.sessions_dir||'?'}</span></div>`;
  });
}

async function deleteSession(sid){
  if(!confirm('Delete session '+sid+'?'))return;
  await api(`/api/v2/sessions/${sid}`,{method:'DELETE'});
  if(currentSession===sid)showPage('sessions');
  else loadSessions();
}

async function retrySession(sid){
  try{await apiJson(`/api/v2/sessions/${sid}/retry`,{method:'POST'});openSession(sid);}
  catch(e){alert('Retry failed: '+e.message);}
}

function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// Legacy inline bootstrap intentionally disabled in favor of /ui/dashboard.js
if(!window.__LVT_USE_EXTERNAL_DASHBOARD__){
  if(TOKEN){
    apiJson('/api/v2/health').then(()=>{
      document.getElementById('auth-gate').classList.add('hidden');
      showPage('dashboard');
    }).catch(()=>showAuthGate());
  } else showAuthGate();
}
</script>
<script src="/ui/dashboard.js"></script>
</body>
</html>"""
