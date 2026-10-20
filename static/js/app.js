/* ── State ───────────────────────────────────── */
let currentMode    = 'supervised';
let currentColumns = [];
let stepsList      = [];

/* ── Init ────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => setMode('supervised'));

/* ── View tabs ───────────────────────────────── */
function showView(v) {
  ['upload','data','results'].forEach(x => {
    document.getElementById('view-' + x).classList.toggle('active', x === v);
    document.getElementById('tab-'  + x).classList.toggle('active', x === v);
  });
  if (v === 'data' && currentColumns.length) refreshTable();
}

/* ── Overlay ─────────────────────────────────── */
function showLoading(msg) {
  document.getElementById('overlay-msg').textContent = msg || 'Processing…';
  document.getElementById('overlay').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('overlay').classList.add('hidden');
}

/* ── Toast ───────────────────────────────────── */
function showToast(msg, ms) {
  const el = document.createElement('div');
  el.textContent = msg;
  Object.assign(el.style, {
    position:'fixed', bottom:'20px', right:'20px',
    background:'#b03030', color:'#fff',
    padding:'10px 16px', borderRadius:'6px',
    fontSize:'12px', zIndex:'9999',
    maxWidth:'380px', fontFamily:'Courier New,monospace',
    lineHeight:'1.5'
  });
  document.body.appendChild(el);
  setTimeout(() => el.remove(), ms || 5000);
}

/* ── Mode switch (no description box) ───────────*/
function setMode(mode) {
  currentMode = mode;
  const sup = mode === 'supervised';

  document.getElementById('tab-sup').className   = 'mode-tab' + (sup   ? ' active-sup'   : '');
  document.getElementById('tab-unsup').className = 'mode-tab' + (!sup  ? ' active-unsup' : '');

  const algoSel = document.getElementById('algo-select');
  algoSel.innerHTML = sup
    ? `<option value="">— select algorithm —</option>
       <optgroup label="Classification">
         <option value="knn">K-Nearest Neighbors</option>
         <option value="dt">Decision Tree</option>
         <option value="nb">Naive Bayes</option>
         <option value="lr">Logistic Regression</option>
         <option value="rfc">Random Forest</option>
         <option value="clapp">CLAPP (Anomaly Detection)</option>
       </optgroup>
       <optgroup label="Regression">
         <option value="linreg">Linear Regression</option>
         <option value="ridge">Ridge Regression</option>
         <option value="rfr">Random Forest Regressor</option>
       </optgroup>`
    : `<option value="">— select algorithm —</option>
       <optgroup label="Clustering">
         <option value="kmeans">K-Means</option>
         <option value="dbscan">DBSCAN</option>
       </optgroup>
       <optgroup label="Dimensionality">
         <option value="pca">PCA</option>
       </optgroup>`;

  document.getElementById('target-row').style.display  = sup ? 'flex'  : 'none';
  document.getElementById('folds-label').style.display = sup ? ''      : 'none';
  document.getElementById('cv-folds').style.display    = sup ? ''      : 'none';
  document.getElementById('k-label').textContent       = sup ? 'k (knn)' : 'k (clusters)';
  
  if (typeof onAlgoChange === 'function') onAlgoChange();
}

function onAlgoChange() {
  const algo = document.getElementById('algo-select').value;
  const clappRow = document.getElementById('clapp-params-row');
  if (clappRow) clappRow.classList.toggle('hidden', algo !== 'clapp');
  document.getElementById('k-label').textContent = algo === 'clapp' ? 'k (knn)' : (currentMode === 'supervised' ? 'k (knn)' : 'k (clusters)');
}

/* ── Drag & drop ─────────────────────────────── */
function onDragOver(e)  { e.preventDefault(); document.getElementById('drop-zone').classList.add('over'); }
function onDragLeave()   { document.getElementById('drop-zone').classList.remove('over'); }
function onDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('over');
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
}
function handleUpload(e) { if (e.target.files[0]) uploadFile(e.target.files[0]); }

/* ── Upload ──────────────────────────────────── */
async function uploadFile(file) {
  showLoading(`Uploading ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)…`);
  const form = new FormData();
  form.append('file', file);
  try {
    const res  = await fetch('/api/upload', { method:'POST', body:form });
    const data = await res.json();
    if (data.error) { showToast(data.error); return; }
    document.getElementById('file-label').textContent =
      `${file.name}  ·  ${data.stats.rows.toLocaleString()} rows`;
    onDataLoaded(data.columns, data.stats);
    stepsList = [];
    renderSteps();
    showView('data');
  } catch(e) { showToast('Upload failed: ' + e.message); }
  finally    { hideLoading(); }
}

async function loadSample() {
  showLoading('Generating sample dataset…');
  const edus = ['Bachelors','Masters','HS-grad','Some-college','Doctorate'];
  const cats  = ['A','B','C'];
  let csv = 'age,income,education,hours_per_week,score,category,target\n';
  for (let i = 0; i < 300; i++) {
    csv += [
      i%7===0  ? '' : Math.floor(Math.random()*45+20),
      i%11===0 ? '' : Math.floor(Math.random()*80000+20000),
      edus[Math.floor(Math.random()*edus.length)],
      i%9===0  ? '' : Math.floor(Math.random()*30+25),
      (Math.random()*100).toFixed(1),
      cats[i%3],
      i%2===0  ? '<=50K' : '>50K'
    ].join(',') + '\n';
  }
  hideLoading();
  await uploadFile(new File([csv], 'sample_300.csv', {type:'text/csv'}));
}

/* ── Columns / stats ─────────────────────────── */
function onDataLoaded(cols, stats) {
  currentColumns = cols;
  const opts = '<option value="">— select —</option>' +
               cols.map(c => `<option value="${c}">${c}</option>`).join('');
  document.getElementById('col-select').innerHTML  = opts;
  document.getElementById('target-col').innerHTML  = opts;
  if (cols.length) document.getElementById('target-col').value = cols[cols.length-1];

  document.getElementById('col-tags-wrap').classList.remove('hidden');
  document.getElementById('col-tags').innerHTML =
    cols.map(c => `<span class="col-tag">${c}</span>`).join('');

  renderStats(stats);
}

function renderStats(s) {
  document.getElementById('stats-bar').innerHTML = `
    <div class="stat"><div class="sv">${s.rows.toLocaleString()}</div><div class="sl">Rows</div></div>
    <div class="stat"><div class="sv">${s.cols}</div><div class="sl">Columns</div></div>
    <div class="stat"><div class="sv">${s.missing.toLocaleString()}</div><div class="sl">Missing</div></div>
    <div class="stat"><div class="sv">${s.dups.toLocaleString()}</div><div class="sl">Duplicates</div></div>`;
}

/* ── Table ───────────────────────────────────── */
async function refreshTable() {
  if (!currentColumns.length) return;
  const target = document.getElementById('target-col').value;
  try {
    const res  = await fetch(`/api/preview?target=${encodeURIComponent(target)}`);
    const data = await res.json();
    if (data.error) return;
    renderStats(data.stats);
    renderTable(data);
  } catch(e) {}
}

function renderTable(data) {
  const wrap = document.getElementById('table-wrap');
  if (!data.rows?.length) { wrap.innerHTML = '<p class="empty-hint">No data</p>'; return; }

  let html = '<table><thead><tr>' +
    data.columns.map(c =>
      `<th class="${c.target?'col-target':''}">${c.name}${c.target?' ★':''}` +
      `<br><span style="font-size:10px;font-weight:400;color:var(--muted)">` +
      `${c.dtype}${c.missing?' · '+c.missing+' null':''}</span></th>`
    ).join('') + '</tr></thead><tbody>';

  data.rows.forEach(r => {
    html += '<tr>' + data.columns.map(c => {
      const v = r[c.name];
      return `<td class="${v===null?'miss':''}">${v===null?'?':v}</td>`;
    }).join('') + '</tr>';
  });
  html += '</tbody></table>';

  if (data.total_rows > data.rows.length)
    html += `<div class="tbl-foot">Showing ${data.rows.length.toLocaleString()} of ` +
            `${data.total_rows.toLocaleString()} rows` +
            (data.truncated_cols ? ` · first ${data.columns.length} columns` : '') + '</div>';

  wrap.innerHTML = html;
}

/* ── Preprocess ──────────────────────────────── */
function onPreprocessChange() {
  const v = document.getElementById('preprocess-select').value;
  document.getElementById('col-select').classList.toggle('hidden', v !== 'drop_col');
}

async function applyPreprocess() {
  const step = document.getElementById('preprocess-select').value;
  const col  = document.getElementById('col-select').value;
  if (!step)                { showToast('Select a preprocessing step'); return; }
  if (!currentColumns.length) { showToast('Load a dataset first');       return; }

  showLoading('Applying ' + step.replace(/_/g,' ') + '…');
  try {
    const res  = await fetch('/api/preprocess', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ step, col })
    });
    const data = await res.json();
    if (data.error) { showToast(data.error); return; }
    onDataLoaded(data.columns, data.stats);
    stepsList.push(data.message);
    renderSteps();
    document.getElementById('preprocess-select').value = '';
    document.getElementById('col-select').classList.add('hidden');
    if (document.getElementById('view-data').classList.contains('active')) refreshTable();
  } catch(e) { showToast(e.message); }
  finally    { hideLoading(); }
}

async function resetData() {
  if (!currentColumns.length) { showToast('No dataset loaded'); return; }
  showLoading('Resetting…');
  try {
    const res  = await fetch('/api/reset', { method:'POST' });
    const data = await res.json();
    if (data.error) { showToast(data.error); return; }
    onDataLoaded(data.columns, data.stats);
    stepsList = [];
    renderSteps();
    if (document.getElementById('view-data').classList.contains('active')) refreshTable();
  } catch(e) { showToast(e.message); }
  finally    { hideLoading(); }
}

function renderSteps() {
  document.getElementById('steps-log').innerHTML = stepsList.length
    ? stepsList.map((s,i) => `<div>${i+1}. ${s}</div>`).join('')
    : 'none';
}

/* ── Run algorithm ───────────────────────────── */
async function runAlgorithm() {
  const algo   = document.getElementById('algo-select').value;
  const target = document.getElementById('target-col').value;
  const k      = Math.max(1, parseInt(document.getElementById('k-param').value) || 3);
  const folds  = Math.max(2, parseInt(document.getElementById('cv-folds').value) || 5);

  if (!algo)                                     { showToast('Select an algorithm'); return; }
  if (!currentColumns.length)                    { showToast('Load a dataset first'); return; }
  if (currentMode==='supervised' && !target)     { showToast('Select a target column'); return; }

  if (algo === 'clapp') {
    return runCLAPP();
  }

  const names = {
    knn:'K-Nearest Neighbors', dt:'Decision Tree', nb:'Naive Bayes',
    lr:'Logistic Regression', rfc:'Random Forest',
    linreg:'Linear Regression', ridge:'Ridge Regression', rfr:'Random Forest (Reg)',
    kmeans:'K-Means', dbscan:'DBSCAN', pca:'PCA'
  };
  showLoading(`Running ${names[algo]||algo}…`);

  try {
    const res  = await fetch('/api/run', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ algo, target, mode:currentMode, k, folds })
    });
    const data = await res.json();
    if (data.error) { showToast(data.error); return; }
    showView('results');
    renderResults(data, names[algo]||algo);
  } catch(e) { showToast(e.message); }
  finally    { hideLoading(); }
}

/* ── Render results ──────────────────────────── */
function tile(val, label, color) {
  return `<div class="tile">
    <div class="tv ${color||''}">${val}</div>
    <div class="tl">${label}</div>
  </div>`;
}

function renderResults(data, algoName) {
  const area  = document.getElementById('results-area');
  const card  = document.createElement('div');
  card.className = 'result-card';

  const isReg   = ['linreg','ridge','rfr'].includes(data.algo);
  const isCls   = ['knn','dt','nb','lr','rfc'].includes(data.algo);
  const isUnsup = data.mode === 'unsupervised';

  let h = `<div class="rc-title">
    ${algoName}
    <span class="rc-badge ${isUnsup?'badge-unsup':'badge-sup'}">${data.mode}</span>
  </div>`;

  /* ── Summary tiles ─────────────────────────── */
  h += '<div class="tiles">';
  h += tile(data.n_samples.toLocaleString(), 'Clean Rows');
  h += tile(data.n_features,                'Features');

  if (isCls && data.summary) {
    h += tile(data.summary['Mean Accuracy'],  'Mean Accuracy',  'green');
    h += tile(data.summary['± Std'],          '± Std');
    h += tile(data.summary['Mean F1'],        'Mean F1',        'green');
    h += tile(data.summary['Mean Precision'], 'Mean Precision');
    h += tile(data.summary['Mean Recall'],    'Mean Recall');
  } else if (isReg && data.summary) {
    h += tile(data.summary['Mean RMSE'],  'Mean RMSE',  'blue');
    h += tile(data.summary['Mean R²'],    'Mean R²',    'blue');
    h += tile(data.summary['Mean MAE'],   'Mean MAE');
    h += tile(data.summary['± Std'],      '± Std RMSE');
  } else if (data.algo === 'kmeans') {
    h += tile(data.silhouette,
              'Silhouette',
              parseFloat(data.silhouette) > 0.5 ? 'green' :
              parseFloat(data.silhouette) > 0.2 ? 'blue'  : 'red');
    h += tile(data.inertia.toLocaleString(), 'Inertia');
  } else if (data.algo === 'dbscan') {
    h += tile(data.n_clusters,     'Clusters', 'purple');
    h += tile(data.noise_pct+'%',  'Noise %');
    h += tile(data.eps,            'Auto eps');
    h += tile(data.min_samples,    'MinPts');
  }
  h += '</div>';

  /* ── Fold table ────────────────────────────── */
  if (data.fold_results?.length) {
    h += '<div class="sub-title">Cross-Validation Folds</div>';
    const keys = Object.keys(data.fold_results[0]).filter(k => k !== 'fold');
    h += '<div style="overflow-x:auto"><table class="dt"><thead><tr>';
    h += '<th>Fold</th>' + keys.map(k=>`<th>${k}</th>`).join('') + '</tr></thead><tbody>';
    data.fold_results.forEach(f => {
      h += `<tr><td>${f.fold}</td>` + keys.map(k=>`<td>${f[k]}</td>`).join('') + '</tr>';
    });
    h += '</tbody></table></div>';

    if (data.summary) {
      h += '<div class="sub-title">Summary</div><div class="tiles">';
      Object.entries(data.summary).forEach(([k,v]) => h += tile(v, k));
      h += '</div>';
    }
  }

  /* ── Full metrics (regression) ─────────────── */
  if (data.full_metrics) {
    h += '<div class="sub-title">Full-Data Metrics</div><div class="tiles">';
    Object.entries(data.full_metrics).forEach(([k,v]) => h += tile(v, k, 'blue'));
    h += '</div>';
  }

  /* ── Per-class report ──────────────────────── */
  if (data.class_report) {
    h += '<div class="sub-title">Per-Class Metrics</div>';
    h += '<div style="overflow-x:auto"><table class="dt"><thead><tr>';
    h += '<th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>';
    h += '</tr></thead><tbody>';
    data.class_report.forEach(r => {
      h += `<tr>
        <td>${r.class}</td>
        <td>${(r.precision*100).toFixed(1)}%</td>
        <td>${(r.recall*100).toFixed(1)}%</td>
        <td>${(r.f1*100).toFixed(1)}%</td>
        <td>${r.support.toLocaleString()}</td>
      </tr>`;
    });
    h += '</tbody></table></div>';
  }

  /* ── Confusion matrix ──────────────────────── */
  if (data.confusion_matrix) {
    const cm = data.confusion_matrix;
    h += '<div class="sub-title">Confusion Matrix' +
         '<span style="font-weight:400;color:var(--muted)"> (rows = actual · cols = predicted)</span></div>';
    h += '<div class="cm-wrap"><table class="cm-tbl"><thead><tr><th></th>';
    cm.labels.forEach(l => h += `<th>pred:${l}</th>`);
    h += '</tr></thead><tbody>';
    cm.matrix.forEach((row, i) => {
      h += `<tr><th>act:${cm.labels[i]}</th>`;
      row.forEach((v,j) => h += `<td class="${i===j?'cm-diag':''}">${v}</td>`);
      h += '</tr>';
    });
    h += '</tbody></table></div>';
  }

  /* ── K-Means ───────────────────────────────── */
  if (data.algo === 'kmeans') {
    if (data.cluster_counts && Object.keys(data.cluster_counts).length) {
      h += '<div class="sub-title">Cluster Sizes</div><div class="c-grid">';
      Object.entries(data.cluster_counts).forEach(([i,cnt]) =>
        h += `<div class="c-card">
          <div class="c-id">Cluster ${+i+1}</div>
          <div class="c-cnt">${cnt.toLocaleString()}</div>
        </div>`);
      h += '</div>';
    }
    if (data.centroids?.length) {
      const ckeys = Object.keys(data.centroids[0]).filter(k=>k!=='cluster'&&k!=='size');
      h += '<div class="sub-title">Centroids (first 6 features)</div>';
      h += '<div style="overflow-x:auto"><table class="dt"><thead><tr>';
      h += '<th>Cluster</th><th>Size</th>' + ckeys.map(k=>`<th>${k}</th>`).join('') + '</tr></thead><tbody>';
      data.centroids.forEach(c =>
        h += `<tr><td>${c.cluster}</td><td>${c.size.toLocaleString()}</td>` +
             ckeys.map(k=>`<td>${c[k]}</td>`).join('') + '</tr>'
      );
      h += '</tbody></table></div>';
    }
  }

  /* ── DBSCAN ────────────────────────────────── */
  if (data.algo === 'dbscan') {
    if (Object.keys(data.cluster_counts||{}).length) {
      h += '<div class="sub-title">Cluster Sizes</div><div class="c-grid">';
      Object.entries(data.cluster_counts).forEach(([i,cnt]) =>
        h += `<div class="c-card">
          <div class="c-id">Cluster ${+i+1}</div>
          <div class="c-cnt">${cnt.toLocaleString()}</div>
        </div>`);
      h += '</div>';
    } else {
      h += '<div style="font-size:12px;color:var(--red);margin-top:8px">All points classified as noise — data may need normalization.</div>';
    }
  }

  /* ── PCA ───────────────────────────────────── */
  if (data.components) {
    h += `<div class="sub-title">Variance Explained
      <span style="font-weight:400;color:var(--muted)">
        · ${data.n_components_90pct} PC(s) needed for 90%
      </span></div>`;
    data.components.forEach(c => {
      h += `<div class="pca-row">
        <span class="pca-lbl">PC${c.pc}</span>
        <div class="pca-bg"><div class="pca-fg" style="width:${Math.min(c.variance_pct,100)}%"></div></div>
        <span class="pca-pct">${c.variance_pct}% <span style="color:var(--hint)">(cum ${c.cumulative_pct}%)</span></span>
      </div>`;
    });
    if (data.loadings?.length) {
      h += '<div class="sub-title">Top Feature Loadings</div>';
      data.loadings.forEach(l => {
        h += `<p style="font-size:12px;margin-bottom:6px"><strong>PC${l.pc}:</strong>&nbsp;`;
        h += l.top_features.map(f =>
          `${f.feature}&nbsp;<span style="color:var(--muted)">(${f.loading>=0?'+':''}${f.loading})</span>`
        ).join('&nbsp;&nbsp;·&nbsp;&nbsp;');
        h += '</p>';
      });
    }
  }

  card.innerHTML = h;
  area.insertBefore(card, area.firstChild);
}

function clearResults() {
  document.getElementById('results-area').innerHTML = '';
}

/* ── CLAPP ───────────────────────────────────── */
async function runCLAPP() {
  const target = document.getElementById('target-col').value;
  const k      = Math.max(1, parseInt(document.getElementById('k-param').value) || 1);
  const folds  = Math.max(2, parseInt(document.getElementById('cv-folds').value) || 5);
  const sigma  = parseFloat(document.getElementById('clapp-sigma').value) || 0.5;
  const thresh = parseFloat(document.getElementById('clapp-thresh').value) || 0.64;

  if (!currentColumns.length) { showToast('Load a dataset first'); return; }
  if (!target)                 { showToast('Select a target column for CLAPP'); return; }

  showLoading('Running CLAPP…');
  try {
    const res  = await fetch('/api/clapp', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ target, k, folds, sigma, threshold: thresh })
    });
    const data = await res.json();
    if (data.error) { showToast(data.error); return; }
    showView('results');
    renderCLAPP(data);
  } catch(e) { showToast(e.message); }
  finally    { hideLoading(); }
}

function renderCLAPP(data) {
  const area = document.getElementById('results-area');
  const card = document.createElement('div');
  card.className = 'result-card clapp-card';
  const d    = data.diagnostics || {};

  let h = `<div class="rc-title">
    CLAPP <span class="rc-badge clapp-badge-inline">Posterior Probability Classifier</span>
  </div>`;

  // ── Overview tiles ──────────────────────────
  h += '<div class="tiles">';
  h += tile(data.n_samples.toLocaleString(), 'Samples');
  h += tile(data.n_features,                'Features');
  h += tile(d.n_clusters || '—',            'Clusters formed', 'purple');
  h += tile((d.dim_reduction || 0) + '%',   'Dim. reduction',  'blue');
  if (data.summary) {
    h += tile(data.summary['Overall Accuracy'], 'Overall Accuracy', 'green');
    h += tile(data.summary['Mean F1 (macro)'],  'Mean F1',          'green');
  }
  h += `<div class="tile"><div class="tv">${d.threshold}</div><div class="tl">Threshold</div></div>`;
  h += `<div class="tile"><div class="tv">${d.sigma}</div><div class="tl">Sigma (σ)</div></div>`;
  h += '</div>';

  // ── Step 1: Posterior Probabilities ──────────
  if (d.posterior_table && d.posterior_table.length) {
    h += clappStep(1, 'Posterior Probability Vectors C(j)',
      "For each feature Fj: Pr(Dd|Fj) = Σ[X(i,j)·Md] / Σ X(i,j) — what fraction of this feature's occurrences belong to each class?");
    const pKeys = Object.keys(d.posterior_table[0]).filter(k => k !== 'feature');
    h += clappTable(['Feature', ...pKeys], d.posterior_table.map(r => [r.feature, ...pKeys.map(k => r[k])]));
  }

  // ── Step 2: Cluster Summary ────────────────
  if (d.cluster_summary && d.cluster_summary.length) {
    h += clappStep(2, 'Self-Constructing Feature Clusters',
      `Fuzzy threshold = ${d.threshold}, σ = ${d.sigma}. Each feature pattern vector was assigned to an existing cluster (if similarity ≥ threshold) or started a new one.`);
    const cKeys = Object.keys(d.cluster_summary[0]).filter(k => !['cluster','members'].includes(k));
    h += clappTable(
      ['Cluster', 'Members', ...cKeys],
      d.cluster_summary.map(r => [r.cluster, r.members.join(', '), ...cKeys.map(k => r[k])])
    );
  }

  // ── Step 3: Transformation Matrix ─────────
  if (d.transform_matrix && d.transform_matrix.length) {
    h += clappStep(3, 'Transformation Matrix T [Features × Clusters]',
      'T[i][g] = fuzzy similarity of pattern vector C(i) to cluster mean M(g). This is the mapping matrix.');
    const tKeys = Object.keys(d.transform_matrix[0]).filter(k => k !== 'feature');
    h += clappTable(['Feature', ...tKeys], d.transform_matrix.map(r => [r.feature, ...tKeys.map(k => r[k])]));
  }

  // ── Step 4: Reduced Matrix ─────────────────
  if (d.reduced_matrix && d.reduced_matrix.length) {
    h += clappStep(4, 'Dimensionality-Reduced Process Matrix [P × C]',
      'Reduced = Original[P×S] × T[S×C]. Each process is now represented by cluster-space values instead of raw features.');
    const rKeys = Object.keys(d.reduced_matrix[0]).filter(k => !['process','label'].includes(k));
    h += clappTable(
      ['Process', ...rKeys, 'Class'],
      d.reduced_matrix.map(r => [r.process, ...rKeys.map(k => r[k]), r.label]),
      d.reduced_matrix.length - 1   // last col is class
    );
  }

  // ── Step 5: CV Folds ──────────────────────
  if (data.fold_results && data.fold_results.length) {
    h += clappStep(5, 'Cross-Validation (kNN on Reduced Matrix)',
      `k = ${document.getElementById('k-param').value}, ${data.fold_results.length}-fold stratified CV`);
    h += clappTable(['Fold','Accuracy','Samples'], data.fold_results.map(f=>[f.fold, f.Accuracy, f.Samples]));
  }

  // ── Summary + Per-class + CM ───────────────
  if (data.summary) {
    h += '<div class="sub-title">Summary</div><div class="tiles">';
    Object.entries(data.summary).forEach(([k,v]) => h += tile(v, k, k.includes('Accuracy')||k.includes('F1')?'green':''));
    h += '</div>';
  }

  if (data.class_report) {
    h += '<div class="sub-title">Per-Class Metrics</div>';
    h += clappTable(
      ['Class','Precision','Recall','F1','Support'],
      data.class_report.map(r => [
        r.class,
        (r.precision*100).toFixed(1)+'%',
        (r.recall*100).toFixed(1)+'%',
        (r.f1*100).toFixed(1)+'%',
        r.support
      ])
    );
  }

  if (data.confusion_matrix) {
    const cm = data.confusion_matrix;
    h += '<div class="sub-title">Confusion Matrix <span style="font-weight:400;color:var(--muted)">(rows=actual · cols=predicted)</span></div>';
    h += '<div class="cm-wrap"><table class="cm-tbl"><thead><tr><th></th>';
    cm.labels.forEach(l => h += `<th>pred:${l}</th>`);
    h += '</tr></thead><tbody>';
    cm.matrix.forEach((row,i) => {
      h += `<tr><th>act:${cm.labels[i]}</th>`;
      row.forEach((v,j) => h += `<td class="${i===j?'cm-diag':''}">${v}</td>`);
      h += '</tr>';
    });
    h += '</tbody></table></div>';
  }

  card.innerHTML = h;
  area.insertBefore(card, area.firstChild);
}

function clappStep(n, title, desc) {
  return `<div class="clapp-step">
    <div class="clapp-step-hdr">
      <span class="clapp-step-num">${n}</span>
      <div>
        <div class="clapp-step-title">${title}</div>
        <div class="clapp-step-desc">${desc}</div>
      </div>
    </div>
  </div>`;
}

function clappTable(headers, rows, colorLastCol) {
  let h = '<div style="overflow-x:auto;margin-bottom:14px"><table class="dt"><thead><tr>';
  h += headers.map(hd => `<th>${hd}</th>`).join('');
  h += '</tr></thead><tbody>';
  rows.forEach(row => {
    h += '<tr>' + row.map((cell, ci) => {
      const isClass = ci === colorLastCol;
      const cls = isClass && String(cell).toUpperCase().includes('NORMAL') && !String(cell).toUpperCase().includes('AB')
                  ? 'style="color:var(--green);font-weight:700"'
                  : isClass ? 'style="color:var(--red);font-weight:700"' : '';
      return `<td ${cls}>${cell}</td>`;
    }).join('') + '</tr>';
  });
  h += '</tbody></table></div>';
  return h;
}
