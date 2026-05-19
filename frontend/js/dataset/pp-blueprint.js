DatasetManagerUI.prototype.setupPreprocessBP = function () {
    document.getElementById('ppClearBtn').addEventListener('click', () => {
        this.ppNodes = [];
        this.ppConns = [];
        this.selectedPpNode = null;
        this.ppIdCounter = 0;
        this.renderPpNodes();
        this.renderPpConns();
    });
    document.getElementById('ppExecBtn').addEventListener('click', () => this.execPreprocess());

    document.querySelectorAll('.pp-palette-item').forEach(item => {
        item.addEventListener('click', () => this.addPpNode(item.dataset.type));
    });

    const canvas = document.getElementById('ppCanvas');
    canvas.addEventListener('mousedown', (e) => this.onPpMouseDown(e));
    window.addEventListener('mousemove', (e) => this.onPpMouseMove(e));
    window.addEventListener('mouseup', () => { this.draggingNode = null; });
};

DatasetManagerUI.prototype.addPpNode = function (type) {
    const id = `pp_${++this.ppIdCounter}`;
    const canvas = document.getElementById('ppCanvas');
    const scrollLeft = canvas.scrollLeft || 0;
    const scrollTop = canvas.scrollTop || 0;
    const x = 30 + (this.ppNodes.length * 190) % 550 + scrollLeft;
    const y = 30 + Math.floor((this.ppNodes.length * 190) / 550) * 130 + scrollTop;
    const node = { id, type, x, y, params: this.defaultParams(type), label: this.nodeLabel(type) };
    this.ppNodes.push(node);
    if (this.ppNodes.length > 1) {
        const prev = this.ppNodes[this.ppNodes.length - 2];
        this.ppConns.push({ from: prev.id, to: id });
    }
    this.renderPpNodes();
    this.renderPpConns();
};

DatasetManagerUI.prototype.defaultParams = function (type) {
    const d = {
        filter_class: { classes: '', mode: 'keep' },
        remove_samples: { count: 100, strategy: 'random' },
        split: { train_ratio: 0.8, val_ratio: 0.2 },
        balance: { method: 'undersample' },
        normalize: { method: 'zscore' },
        resize: { width: 224, height: 224 },
        one_hot: { columns: '', drop_first: false, max_categories: 50 },
        label_encode: { columns: '', sort_by_freq: false },
        ordinal_encode: { columns: '', mappings: '' },
        target_encode: { columns: '', label_col: '', smoothing: 1.0 },
        frequency_encode: { columns: '' },
        binary_encode: { columns: '' },
        hash_encode: { columns: '', n_components: 8, signed: false },
        purge_all: {}
    };
    return d[type] || {};
};

DatasetManagerUI.prototype.nodeLabel = function (type) {
    const l = {
        filter_class: 'Filter Class', remove_samples: 'Remove Samples',
        split: 'Split Dataset', balance: 'Balance Classes',
        normalize: 'Normalize', resize: 'Resize Images',
        one_hot: 'One-Hot Encode', label_encode: 'Label Encode',
        ordinal_encode: 'Ordinal Encode', target_encode: 'Target Encode',
        frequency_encode: 'Frequency Encode', binary_encode: 'Binary Encode',
        hash_encode: 'Hash Encode', purge_all: 'Purge All Data'
    };
    return l[type] || type;
};

DatasetManagerUI.prototype.renderPpNodes = function () {
    const c = document.getElementById('ppNodes');
    c.innerHTML = this.ppNodes.map(n => `
        <div class="bp-node ${this.selectedPpNode === n.id ? 'selected' : ''}" data-id="${n.id}" style="left:${n.x}px;top:${n.y}px;">
            <button class="bp-node-del" data-id="${n.id}"><svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 2l6 6M8 2l-6 6" stroke="currentColor" stroke-width="1.5"/></svg></button>
            <div class="bp-node-header ${n.type}"><svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">${this.nodeIcon(n.type)}</svg>${n.label}</div>
            <div class="bp-node-body">${this.nodeSummary(n)}</div>
            <div class="bp-node-ports"><div class="bp-port in"></div><div class="bp-port out"></div></div>
        </div>
    `).join('');

    c.querySelectorAll('.bp-node').forEach(el => {
        el.addEventListener('dblclick', (e) => { if (!e.target.closest('.bp-node-del')) this.openNodeConfig(el.dataset.id); });
    });
    c.querySelectorAll('.bp-node-del').forEach(btn => {
        btn.addEventListener('click', (e) => { e.stopPropagation(); this.removePpNode(btn.dataset.id); });
    });
};

DatasetManagerUI.prototype.nodeIcon = function (type) {
    const i = {
        filter_class: '<path d="M2 3h8l-3 4v3l-2 1V7L2 3z" fill="none" stroke="currentColor" stroke-width="1.5"/>',
        remove_samples: '<circle cx="3.5" cy="3.5" r="2" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="8.5" cy="8.5" r="2" fill="none" stroke="currentColor" stroke-width="1.5"/><line x1="5" y1="5" x2="7" y2="7" stroke="currentColor" stroke-width="1.5"/>',
        split: '<rect x="1" y="1" width="4" height="4" rx="0.5" fill="none" stroke="currentColor" stroke-width="1.2"/><rect x="7" y="1" width="4" height="4" rx="0.5" fill="none" stroke="currentColor" stroke-width="1.2"/><rect x="1" y="7" width="4" height="4" rx="0.5" fill="none" stroke="currentColor" stroke-width="1.2"/><rect x="7" y="7" width="4" height="4" rx="0.5" fill="none" stroke="currentColor" stroke-width="1.2"/>',
        balance: '<path d="M6 1v10M2 3.5h8M2 8.5h8" stroke="currentColor" stroke-width="1.5"/>',
        normalize: '<path d="M1 11 L4 2 L7 7 L11 1" fill="none" stroke="currentColor" stroke-width="1.5"/>',
        resize: '<rect x="1" y="1" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M9 1v2.5h2.5" fill="none" stroke="currentColor" stroke-width="1.5"/>',
        one_hot: '<rect x="1" y="1" width="3" height="3" rx="0.5" fill="currentColor"/><rect x="5.5" y="1" width="3" height="3" rx="0.5" fill="none" stroke="currentColor" stroke-width="1"/><rect x="1" y="5.5" width="3" height="3" rx="0.5" fill="none" stroke="currentColor" stroke-width="1"/><rect x="5.5" y="5.5" width="3" height="3" rx="0.5" fill="currentColor"/>',
        label_encode: '<circle cx="3.5" cy="3.5" r="2" fill="currentColor"/><circle cx="8.5" cy="8.5" r="2" fill="currentColor"/><line x1="5" y1="5" x2="7" y2="7" stroke="currentColor" stroke-width="1.5"/>',
        ordinal_encode: '<rect x="1" y="1" width="10" height="2.5" rx="0.5" fill="currentColor"/><rect x="1" y="5" width="7" height="2.5" rx="0.5" fill="currentColor" opacity="0.7"/><rect x="1" y="9" width="4" height="2.5" rx="0.5" fill="currentColor" opacity="0.4"/>',
        target_encode: '<circle cx="6" cy="6" r="5" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="6" cy="6" r="2.5" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="6" cy="6" r="0.8" fill="currentColor"/>',
        frequency_encode: '<rect x="1" y="7" width="3" height="4" rx="0.5" fill="currentColor" opacity="0.4"/><rect x="5" y="4" width="3" height="7" rx="0.5" fill="currentColor" opacity="0.6"/><rect x="9" y="1" width="3" height="10" rx="0.5" fill="currentColor"/>',
        binary_encode: '<text x="1" y="5.5" font-size="5.5" fill="currentColor" font-family="monospace">0</text><text x="6" y="5.5" font-size="5.5" fill="currentColor" font-family="monospace">1</text><text x="1" y="11" font-size="5.5" fill="currentColor" font-family="monospace">1</text><text x="6" y="11" font-size="5.5" fill="currentColor" font-family="monospace">0</text>',
        hash_encode: '<path d="M3.5 1v10M8.5 1v10M1 4.5h10M1 8h10" stroke="currentColor" stroke-width="1.5"/>',
        purge_all: '<path d="M2 3.5h8M4 3.5V2h4v1.5M4.5 5.5v4M7.5 5.5v4M3 3.5l.8 6.5h4.4l.8-6.5" fill="none" stroke="currentColor" stroke-width="1.5"/>'
    };
    return i[type] || '';
};

DatasetManagerUI.prototype.nodeSummary = function (n) {
    const p = n.params;
    switch (n.type) {
        case 'filter_class': return `${p.mode} [${p.classes || 'all'}]`;
        case 'remove_samples': return `${p.count} | ${p.strategy}`;
        case 'split': return `Train: ${p.train_ratio} Val: ${p.val_ratio}`;
        case 'balance': return p.method;
        case 'normalize': return p.method;
        case 'resize': return `${p.width}x${p.height}`;
        case 'one_hot': return `Cols: ${p.columns || 'auto'}${p.drop_first ? ' (drop_first)' : ''}`;
        case 'label_encode': return `Cols: ${p.columns || 'auto'}`;
        case 'ordinal_encode': return `Cols: ${p.columns || 'auto'}`;
        case 'target_encode': return `Cols: ${p.columns || 'auto'} | Label: ${p.label_col || '?'}`;
        case 'frequency_encode': return `Cols: ${p.columns || 'auto'}`;
        case 'binary_encode': return `Cols: ${p.columns || 'auto'}`;
        case 'hash_encode': return `Cols: ${p.columns || 'auto'} | ${p.n_components} bins`;
        case 'purge_all': return 'Removes all data';
        default: return '';
    }
};

DatasetManagerUI.prototype.renderPpConns = function () {
    const svg = document.getElementById('ppConns');
    const canvas = document.getElementById('ppCanvas');
    const w = Math.max(canvas.scrollWidth, canvas.clientWidth);
    const h = Math.max(canvas.scrollHeight, canvas.clientHeight);
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);
    svg.innerHTML = '';
    if (!svg.querySelector('defs')) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = '<marker id="ppArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="var(--accent-primary)" opacity="0.6"/></marker>';
        svg.appendChild(defs);
    }
    this.ppConns.forEach(conn => {
        const fn = this.ppNodes.find(n => n.id === conn.from);
        const tn = this.ppNodes.find(n => n.id === conn.to);
        if (!fn || !tn) return;
        const x1 = fn.x + 160, y1 = fn.y + 28;
        const x2 = tn.x, y2 = tn.y + 28;
        const cx = Math.max(Math.abs(x2 - x1) * 0.5, 40);
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M${x1},${y1} C${x1+cx},${y1} ${x2-cx},${y2} ${x2},${y2}`);
        path.setAttribute('marker-end', 'url(#ppArrow)');
        svg.appendChild(path);
    });
};

DatasetManagerUI.prototype.onPpMouseDown = function (e) {
    if (e.target.closest('.bp-node-del')) return;
    const nodeEl = e.target.closest('.bp-node');
    if (nodeEl) {
        const rect = nodeEl.getBoundingClientRect();
        this.dragOff = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        this.selectedPpNode = nodeEl.dataset.id;
        this.draggingNode = this.ppNodes.find(n => n.id === nodeEl.dataset.id);
        this.renderPpNodes();
        e.preventDefault();
    } else {
        this.selectedPpNode = null;
        this.renderPpNodes();
    }
};

DatasetManagerUI.prototype.onPpMouseMove = function (e) {
    if (!this.draggingNode) return;
    const container = document.getElementById('ppNodes');
    const cr = container.getBoundingClientRect();
    this.draggingNode.x = Math.max(0, e.clientX - cr.left - this.dragOff.x);
    this.draggingNode.y = Math.max(0, e.clientY - cr.top - this.dragOff.y);
    this.renderPpNodes();
    this.renderPpConns();
};

DatasetManagerUI.prototype.removePpNode = function (id) {
    this.ppNodes = this.ppNodes.filter(n => n.id !== id);
    this.ppConns = this.ppConns.filter(c => c.from !== id && c.to !== id);
    this.selectedPpNode = null;
    this.renderPpNodes();
    this.renderPpConns();
};

DatasetManagerUI.prototype.openNodeConfig = function (nodeId) {
    const node = this.ppNodes.find(n => n.id === nodeId);
    if (!node) return;
    const modal = document.getElementById('ppNodeModal');
    document.getElementById('ppNodeTitle').textContent = `Configure: ${node.label}`;
    document.getElementById('ppNodeConfig').innerHTML = this.buildNodeForm(node);
    const saveBtn = document.getElementById('ppNodeConfig').querySelector('.save-node-cfg');
    if (saveBtn) saveBtn.addEventListener('click', () => { this.saveNodeCfg(node); modal.style.display = 'none'; });
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
};

DatasetManagerUI.prototype.buildNodeForm = function (node) {
    const p = node.params;
    let h = '';
    switch (node.type) {
        case 'filter_class':
            h = `<div class="property-row"><label class="property-label">Mode</label><select class="property-select" id="pnMode"><option value="keep" ${p.mode==='keep'?'selected':''}>Keep</option><option value="remove" ${p.mode==='remove'?'selected':''}>Remove</option></select></div><div class="property-row"><label class="property-label">Classes (comma-sep)</label><input type="text" class="property-input" id="pnClasses" value="${p.classes}" placeholder="class1, class2"></div>`;
            break;
        case 'remove_samples':
            h = `<div class="property-row"><label class="property-label">Count</label><input type="number" class="property-input" id="pnCount" value="${p.count}" min="1"></div><div class="property-row"><label class="property-label">Strategy</label><select class="property-select" id="pnStrategy"><option value="random" ${p.strategy==='random'?'selected':''}>Random</option><option value="first" ${p.strategy==='first'?'selected':''}>First N</option><option value="last" ${p.strategy==='last'?'selected':''}>Last N</option></select></div>`;
            break;
        case 'split':
            h = `<div class="property-row"><label class="property-label">Train Ratio</label><input type="number" class="property-input" id="pnTrain" value="${p.train_ratio}" min="0.1" max="0.9" step="0.1"></div><div class="property-row"><label class="property-label">Val Ratio</label><input type="number" class="property-input" id="pnVal" value="${p.val_ratio}" min="0.1" max="0.9" step="0.1"></div>`;
            break;
        case 'balance':
            h = `<div class="property-row"><label class="property-label">Method</label><select class="property-select" id="pnBalMethod"><option value="undersample" ${p.method==='undersample'?'selected':''}>Undersample</option><option value="oversample" ${p.method==='oversample'?'selected':''}>Oversample</option></select></div>`;
            break;
        case 'normalize':
            h = `<div class="property-row"><label class="property-label">Method</label><select class="property-select" id="pnNormMethod"><option value="zscore" ${p.method==='zscore'?'selected':''}>Z-Score</option><option value="minmax" ${p.method==='minmax'?'selected':''}>Min-Max</option></select></div>`;
            break;
        case 'resize':
            h = `<div class="property-row"><label class="property-label">Width</label><input type="number" class="property-input" id="pnW" value="${p.width}" min="1"></div><div class="property-row"><label class="property-label">Height</label><input type="number" class="property-input" id="pnH" value="${p.height}" min="1"></div>`;
            break;
        case 'one_hot':
            h = `<div class="property-row"><label class="property-label">Columns (comma-sep)</label><input type="text" class="property-input" id="pnOHCol" value="${p.columns}" placeholder="auto = all categorical"></div><div class="property-row"><label class="property-label">Max Categories</label><input type="number" class="property-input" id="pnOHMax" value="${p.max_categories}" min="2"></div><div class="property-row"><label class="property-checkbox"><input type="checkbox" id="pnOHDrop" ${p.drop_first?'checked':''}> Drop first column (avoid multicollinearity)</label></div>`;
            break;
        case 'label_encode':
            h = `<div class="property-row"><label class="property-label">Columns (comma-sep)</label><input type="text" class="property-input" id="pnLECol" value="${p.columns}" placeholder="auto = all categorical"></div><div class="property-row"><label class="property-checkbox"><input type="checkbox" id="pnLEFreq" ${p.sort_by_freq?'checked':''}> Sort by frequency</label></div>`;
            break;
        case 'ordinal_encode':
            h = `<div class="property-row"><label class="property-label">Columns (comma-sep)</label><input type="text" class="property-input" id="pnOECol" value="${p.columns}" placeholder="auto = all categorical"></div><div class="property-row"><label class="property-label">Mappings (JSON)</label><textarea class="property-input" id="pnOEMap" rows="4" placeholder='{"col1": ["low","mid","high"]}' style="resize:vertical;font-family:monospace;font-size:11px;">${p.mappings}</textarea></div>`;
            break;
        case 'target_encode':
            h = `<div class="property-row"><label class="property-label">Columns (comma-sep)</label><input type="text" class="property-input" id="pnTECol" value="${p.columns}" placeholder="categorical columns"></div><div class="property-row"><label class="property-label">Target Column</label><input type="text" class="property-input" id="pnTELabel" value="${p.label_col}" placeholder="label column name"></div><div class="property-row"><label class="property-label">Smoothing</label><input type="number" class="property-input" id="pnTESmooth" value="${p.smoothing}" min="0" step="0.1"></div>`;
            break;
        case 'frequency_encode':
            h = `<div class="property-row"><label class="property-label">Columns (comma-sep)</label><input type="text" class="property-input" id="pnFECol" value="${p.columns}" placeholder="auto = all categorical"></div>`;
            break;
        case 'binary_encode':
            h = `<div class="property-row"><label class="property-label">Columns (comma-sep)</label><input type="text" class="property-input" id="pnBECol" value="${p.columns}" placeholder="auto = all categorical"></div>`;
            break;
        case 'hash_encode':
            h = `<div class="property-row"><label class="property-label">Columns (comma-sep)</label><input type="text" class="property-input" id="pnHECol" value="${p.columns}" placeholder="auto = all categorical"></div><div class="property-row"><label class="property-label">Hash Components</label><input type="number" class="property-input" id="pnHEBins" value="${p.n_components}" min="2" max="128"></div><div class="property-row"><label class="property-checkbox"><input type="checkbox" id="pnHESigned" ${p.signed?'checked':''}> Signed hashing</label></div>`;
            break;
        case 'purge_all':
            h = '<p style="color:var(--accent-danger);font-size:12px;">Permanently deletes all data. Cannot be undone.</p>';
            break;
    }
    h += '<button class="btn btn-primary btn-full save-node-cfg">Save</button>';
    return h;
};

DatasetManagerUI.prototype.saveNodeCfg = function (node) {
    switch (node.type) {
        case 'filter_class': node.params.mode = document.getElementById('pnMode').value; node.params.classes = document.getElementById('pnClasses').value; break;
        case 'remove_samples': node.params.count = parseInt(document.getElementById('pnCount').value) || 100; node.params.strategy = document.getElementById('pnStrategy').value; break;
        case 'split': node.params.train_ratio = parseFloat(document.getElementById('pnTrain').value) || 0.8; node.params.val_ratio = parseFloat(document.getElementById('pnVal').value) || 0.2; break;
        case 'balance': node.params.method = document.getElementById('pnBalMethod').value; break;
        case 'normalize': node.params.method = document.getElementById('pnNormMethod').value; break;
        case 'resize': node.params.width = parseInt(document.getElementById('pnW').value) || 224; node.params.height = parseInt(document.getElementById('pnH').value) || 224; break;
        case 'one_hot': node.params.columns = document.getElementById('pnOHCol').value; node.params.max_categories = parseInt(document.getElementById('pnOHMax').value) || 50; node.params.drop_first = document.getElementById('pnOHDrop').checked; break;
        case 'label_encode': node.params.columns = document.getElementById('pnLECol').value; node.params.sort_by_freq = document.getElementById('pnLEFreq').checked; break;
        case 'ordinal_encode': node.params.columns = document.getElementById('pnOECol').value; node.params.mappings = document.getElementById('pnOEMap').value; break;
        case 'target_encode': node.params.columns = document.getElementById('pnTECol').value; node.params.label_col = document.getElementById('pnTELabel').value; node.params.smoothing = parseFloat(document.getElementById('pnTESmooth').value) || 1.0; break;
        case 'frequency_encode': node.params.columns = document.getElementById('pnFECol').value; break;
        case 'binary_encode': node.params.columns = document.getElementById('pnBECol').value; break;
        case 'hash_encode': node.params.columns = document.getElementById('pnHECol').value; node.params.n_components = parseInt(document.getElementById('pnHEBins').value) || 8; node.params.signed = document.getElementById('pnHESigned').checked; break;
    }
    this.renderPpNodes();
    this.renderPpConns();
};

DatasetManagerUI.prototype.execPreprocess = async function () {
    if (this.ppNodes.length === 0) { window.app.showToast('Add at least one operation', 'warning'); return; }
    const ds = this.datasets.find(d => d.id === this.selectedId);
    if (!ds) { window.app.showToast('Select a dataset first', 'warning'); return; }
    if (this.ppNodes.some(n => n.type === 'purge_all') && !confirm('Purge ALL data? Continue?')) return;

    const modal = document.getElementById('ppResultModal');
    const content = document.getElementById('ppResultContent');
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    content.innerHTML = '<div class="ds-loading">Executing preprocessing pipeline...</div>';

    try {
        const res = await fetch(`${DS_API}/datasets/${ds.id}/preprocess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(this.ppNodes)
        });
        
        const r = await res.json();
        
        if (r.valid) {
            const newId = r.new_dataset_id;
            let h = `<div class="pp-result-summary">
                <div class="pp-result-item"><span class="pp-result-value">${this.ppNodes.length}</span><span class="pp-result-label">Operations</span></div>
                <div class="pp-result-item"><span class="pp-result-value">0</span><span class="pp-result-label">Errors</span></div>
                <div class="pp-result-item"><span class="pp-result-value">${r.affected_samples.toLocaleString()}</span><span class="pp-result-label">Affected Samples</span></div>
                <div class="pp-result-item"><span class="pp-result-value">${this.ppNodes.length}</span><span class="pp-result-label">Total Ops</span></div>
            </div><div class="pp-log">`;
            h += `<div class="pp-log-entry success">${r.message}</div>`;
            h += `</div><div class="pp-result-new-dataset">`;
            h += `<p>New dataset ID: <strong>${newId}</strong></p>`;
            h += `<button id="ppViewNewBtn" class="btn btn-primary">View Preprocessed Dataset</button>`;
            h += '</div>';
            content.innerHTML = h;
            
            document.getElementById('ppViewNewBtn').addEventListener('click', () => {
                this.selectedId = newId;
                modal.style.display = 'none';
                this.loadDatasets();
                this.selectDataset(newId);
            });
        } else {
            let h = `<div class="pp-result-summary">
                <div class="pp-result-item"><span class="pp-result-value">0</span><span class="pp-result-label">Completed</span></div>
                <div class="pp-result-item"><span class="pp-result-value">${this.ppNodes.length}</span><span class="pp-result-label">Errors</span></div>
            </div><div class="pp-log">`;
            h += `<div class="pp-log-entry error">Preprocessing failed: ${r.errors.join(', ')}</div>`;
            h += '</div>';
            content.innerHTML = h;
        }
    } catch (e) {
        let h = `<div class="pp-result-summary">
            <div class="pp-result-item"><span class="pp-result-value">0</span><span class="pp-result-label">Completed</span></div>
            <div class="pp-result-item"><span class="pp-result-value">1</span><span class="pp-result-label">Errors</span></div>
        </div><div class="pp-log">`;
        h += `<div class="pp-log-entry error">Backend error: ${e.message}</div>`;
        h += '</div>';
        content.innerHTML = h;
    }
};

DatasetManagerUI.prototype.execNode = async function (node, ds) {
    switch (node.type) {
        case 'purge_all': {
            const res = await fetch(`${DS_API}/datasets/${ds.id}`, { method: 'DELETE' });
            const r = await res.json();
            if (r.valid) { this.selectedId = null; document.getElementById('dsWelcome').style.display = ''; document.getElementById('dsView').style.display = 'none'; return { message: 'All data purged', affected: ds.num_samples }; }
            throw new Error(r.errors?.join(', ') || 'Failed');
        }
        default:
            throw new Error(`Node ${node.type} must be executed via backend pipeline`);
    }
};
