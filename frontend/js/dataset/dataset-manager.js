const DS_API = 'http://localhost:8000';

class DatasetManagerUI {
    constructor() {
        this.datasets = [];
        this.selectedId = null;
        this.currentView = 'overview';
        this.ppNodes = [];
        this.ppConns = [];
        this.selectedPpNode = null;
        this.ppIdCounter = 0;
        this.draggingNode = null;
        this.dragOff = { x: 0, y: 0 };
    }
}

DatasetManagerUI.prototype.init = function () {
    this.setupSidebar();
    this.setupViewTabs();
    this.setupActions();
    this.setupPreprocessBP();
    this.loadDatasets();
};

DatasetManagerUI.prototype.setupSidebar = function () {
    document.querySelectorAll('.ds-upload-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const t = e.target.dataset.tab;
            document.querySelectorAll('.ds-upload-tab').forEach(x => x.classList.remove('active'));
            e.target.classList.add('active');
            document.getElementById('dsTabFolder').classList.toggle('active', t === 'folder');
            if (t === 'file') {
                this.selectedId = null;
                this.renderList();
                document.getElementById('dsWelcome').style.display = 'none';
                document.getElementById('dsView').style.display = 'none';
                document.getElementById('dsUploadPanel').style.display = '';
            } else {
                document.getElementById('dsUploadPanel').style.display = 'none';
                if (!this.selectedId) {
                    document.getElementById('dsWelcome').style.display = '';
                } else {
                    document.getElementById('dsView').style.display = '';
                }
            }
        });
    });

    document.getElementById('dsLoadFolderBtn').addEventListener('click', () => this.loadFolder());
    document.getElementById('dsBrowseBtn').addEventListener('click', () => this.browseFolder());

    const dz = document.getElementById('dsUploadDropZone');
    dz.addEventListener('click', () => document.getElementById('dsUploadFileInput').click());
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
    dz.addEventListener('drop', (e) => this.handleUploadDrop(e));
    document.getElementById('dsUploadFileInput').addEventListener('change', (e) => this.handleUploadFile(e));
};

DatasetManagerUI.prototype.setupViewTabs = function () {
    document.querySelectorAll('.ds-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            this.currentView = e.target.dataset.view;
            if (this.currentView === 'overview') {
                this.currentView = 'preview';
                e.target.dataset.view = 'preview';
            }
            document.querySelectorAll('.ds-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            document.querySelectorAll('.ds-panel').forEach(p => p.classList.remove('active'));
            const panelId = 'dsPanel' + this.currentView.charAt(0).toUpperCase() + this.currentView.slice(1);
            const panel = document.getElementById(panelId);
            if (panel) panel.classList.add('active');
            if (this.currentView === 'preview') this.showPreview();
            if (this.currentView === 'visualize') this.showViz();
        });
    });
};

DatasetManagerUI.prototype.setupActions = function () {
    const dsBtn = document.getElementById('datasetBtn');
    if (dsBtn) dsBtn.addEventListener('click', () => this.openModal());

    const closeBtn = document.getElementById('closeDatasetModal');
    if (closeBtn) closeBtn.addEventListener('click', () => this.closeModal());

    const purgeAllBtn = document.getElementById('purgeDatasetsBtn');
    if (purgeAllBtn) purgeAllBtn.addEventListener('click', () => this.purgeAll());

    const delBtn = document.getElementById('dsDeleteBtn');
    if (delBtn) delBtn.addEventListener('click', () => this.deleteDataset());

    document.getElementById('closePpNodeModal').addEventListener('click', () => document.getElementById('ppNodeModal').style.display = 'none');
    document.getElementById('closePpResultModal').addEventListener('click', () => document.getElementById('ppResultModal').style.display = 'none');
};

DatasetManagerUI.prototype.openModal = function () {
    const modal = document.getElementById('datasetModal');
    if (modal) {
        modal.classList.add('active');
        document.querySelectorAll('.ds-upload-tab').forEach(x => x.classList.remove('active'));
        document.querySelector('.ds-upload-tab[data-tab="folder"]').classList.add('active');
        document.getElementById('dsTabFolder').classList.add('active');
        document.getElementById('dsUploadPanel').style.display = 'none';
        if (!this.selectedId) {
            document.getElementById('dsWelcome').style.display = '';
            document.getElementById('dsView').style.display = 'none';
        }
        this.loadDatasets();
    }
};

DatasetManagerUI.prototype.closeModal = function () {
    document.getElementById('datasetModal').classList.remove('active');
    this.selectedId = null;
};

DatasetManagerUI.prototype.loadDatasets = async function () {
    try {
        const res = await fetch(`${DS_API}/datasets`);
        const data = await res.json();
        this.datasets = data.datasets || [];
        this.renderList();
    } catch (e) {
        this.datasets = [];
        this.renderList();
    }
};

DatasetManagerUI.prototype.renderList = function () {
    const list = document.getElementById('dsList');
    const count = document.getElementById('dsCount');
    if (count) count.textContent = this.datasets.length;

    if (this.datasets.length === 0) {
        list.innerHTML = '<div class="ds-empty"><svg width="28" height="28" viewBox="0 0 28 28" fill="none"><ellipse cx="14" cy="7" rx="9" ry="3.5" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"/><path d="M5 7v5c0 2 4 3.5 9 3.5s9-1.5 9-3.5V7" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"/></svg><p>No datasets</p></div>';
        return;
    }

    list.innerHTML = this.datasets.map(ds => `
        <div class="ds-card ${this.selectedId === ds.id ? 'active' : ''}" data-id="${ds.id}">
            <div class="ds-card-icon">
                ${ds.dataset_type.includes('image') ? '<svg width="16" height="16" viewBox="0 0 16 16"><rect x="2" y="2" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="5" cy="5" r="1.5" fill="currentColor"/><path d="M14 11l-4-4-6 6" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>' : '<svg width="16" height="16" viewBox="0 0 16 16"><rect x="2" y="2" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><line x1="5" y1="5" x2="11" y2="5" stroke="currentColor" stroke-width="1.5"/><line x1="5" y1="8" x2="11" y2="8" stroke="currentColor" stroke-width="1.5"/><line x1="5" y1="11" x2="9" y2="11" stroke="currentColor" stroke-width="1.5"/></svg>'}
            </div>
            <div class="ds-card-info">
                <div class="ds-card-name">${ds.name}</div>
                <div class="ds-card-meta">${ds.num_samples.toLocaleString()} samples${ds.num_classes > 0 ? ` · ${ds.num_classes} classes` : ''}</div>
            </div>
            <button class="ds-card-del" data-id="${ds.id}">
                <svg width="10" height="10" viewBox="0 0 10 10"><path d="M3.5 3.5A.5.5 0 014 4v4a.5.5 0 01-1 0V4a.5.5 0 01.5-.5zm2 0a.5.5 0 01.5.5v4a.5.5 0 01-1 0V4a.5.5 0 01.5-.5zm2.5.5a.5.5 0 00-1 0v4a.5.5 0 001 0V4z"/></svg>
            </button>
        </div>
    `).join('');

    list.querySelectorAll('.ds-card').forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.closest('.ds-card-del')) return;
            this.selectDataset(card.dataset.id);
        });
    });

    list.querySelectorAll('.ds-card-del').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectedId = btn.dataset.id;
            this.deleteDataset();
        });
    });
};

DatasetManagerUI.prototype.selectDataset = async function (id) {
    this.selectedId = id;
    this.renderList();
    const ds = this.datasets.find(d => d.id === id);
    if (!ds) return;

    document.querySelectorAll('.ds-upload-tab').forEach(x => x.classList.remove('active'));
    document.querySelector('.ds-upload-tab[data-tab="folder"]').classList.add('active');
    document.getElementById('dsTabFolder').classList.add('active');
    document.getElementById('dsUploadPanel').style.display = 'none';
    document.getElementById('dsWelcome').style.display = 'none';
    document.getElementById('dsView').style.display = '';
    document.getElementById('dsViewName').textContent = ds.name;

    const badges = document.getElementById('dsViewBadges');
    badges.innerHTML = `
        <span class="ds-badge type">${this.fmtType(ds.dataset_type)}</span>
        <span class="ds-badge samples">${ds.num_samples.toLocaleString()} samples</span>
        ${ds.num_classes > 0 ? `<span class="ds-badge classes">${ds.num_classes} classes</span>` : ''}
        <span class="ds-badge shape">[${ds.input_shape.join(', ')}]</span>
    `;

    this.renderOverview(ds);
    this.currentView = 'preview';
    document.querySelectorAll('.ds-tab').forEach(t => t.classList.remove('active'));
    document.querySelector('.ds-tab[data-view="preview"]').classList.add('active');
    document.querySelectorAll('.ds-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('dsPanelPreview').classList.add('active');
};

DatasetManagerUI.prototype.renderOverview = function (ds) {
    const panel = document.getElementById('dsInfoPanel');
    const currentLabel = ds.metadata?.label_column || null;
    const isCsv = ds.dataset_type === 'tabular_csv';

    panel.innerHTML = `
        <div class="ds-info-card"><div class="ds-info-label">Type</div><div class="ds-info-value text">${this.fmtType(ds.dataset_type)}</div></div>
        <div class="ds-info-card"><div class="ds-info-label">Samples</div><div class="ds-info-value">${ds.num_samples.toLocaleString()}</div></div>
        ${ds.num_classes > 0 ? '<div class="ds-info-card"><div class="ds-info-label">Classes</div><div class="ds-info-value">' + ds.num_classes + '</div></div>' : ''}
        <div class="ds-info-card"><div class="ds-info-label">Shape</div><div class="ds-info-value">[${ds.input_shape.join(', ')}]</div></div>
        <div class="ds-info-card"><div class="ds-info-label">Size</div><div class="ds-info-value">${this.fmtSize(ds.file_size)}</div></div>
        <div class="ds-info-card"><div class="ds-info-label">Created</div><div class="ds-info-value text">${new Date(ds.created_at).toLocaleString()}</div></div>
        ${isCsv ? `
        <div class="ds-info-card" style="grid-column:1/-1">
            <div class="ds-info-label">Target Column</div>
            <div class="ds-target-selector">
                <select id="dsTargetSelect" class="ds-target-select">
                    <option value="">— Select target —</option>
                    ${currentLabel ? `<option value="${currentLabel}" selected>${currentLabel} (current)</option>` : ''}
                </select>
                <button id="dsTargetApply" class="ds-target-apply-btn">Apply</button>
                <span id="dsTargetStatus" class="ds-target-status"></span>
            </div>
        </div>
        ` : ''}
        ${ds.class_names && ds.class_names.length > 0 ? '<div class="ds-info-card" style="grid-column:1/-1"><div class="ds-info-label">Classes</div><div class="ds-info-value text">' + ds.class_names.join(', ') + '</div></div>' : ''}
        ${ds.split_info && Object.keys(ds.split_info).length > 0 ? '<div class="ds-info-card" style="grid-column:1/-1"><div class="ds-info-label">Split</div><div class="ds-info-value text">' + Object.entries(ds.split_info).map(([k,v]) => k+': '+v).join(' · ') + '</div></div>' : ''}
    `;

    if (isCsv) {
        this.loadTargetColumns();
        document.getElementById('dsTargetApply').addEventListener('click', () => this.applyTargetColumn());
    }
};

DatasetManagerUI.prototype.loadTargetColumns = async function () {
    const sel = document.getElementById('dsTargetSelect');
    if (!sel || !this.selectedId) return;

    try {
        const res = await fetch(`${DS_API}/datasets/${this.selectedId}/target-columns`);
        const r = await res.json();
        if (!r.valid) return;

        const currentLabel = r.current_label;
        sel.innerHTML = '<option value="">— Select target —</option>';
        r.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col.name;
            opt.textContent = `${col.name} (${col.type}, ${col.unique_count} unique)`;
            opt.dataset.type = col.type;
            opt.dataset.suitable = col.suitable_as_target;
            if (col.name === currentLabel) opt.selected = true;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load target columns:', e);
    }
};

DatasetManagerUI.prototype.applyTargetColumn = async function () {
    const sel = document.getElementById('dsTargetSelect');
    const status = document.getElementById('dsTargetStatus');
    const column = sel?.value;
    if (!column || !this.selectedId) return;

    const btn = document.getElementById('dsTargetApply');
    btn.disabled = true;
    btn.textContent = 'Applying...';
    status.textContent = '';
    status.className = 'ds-target-status';

    try {
        const res = await fetch(`${DS_API}/datasets/${this.selectedId}/label-column?label_column=${encodeURIComponent(column)}`, {
            method: 'PUT'
        });
        const r = await res.json();
        if (r.valid) {
            status.textContent = `✓ ${r.message} (${r.num_classes} classes)`;
            status.className = 'ds-target-status success';

            const ds = this.datasets.find(d => d.id === this.selectedId);
            if (ds) {
                ds.num_classes = r.num_classes;
                ds.class_names = r.class_names;
                ds.input_shape = r.input_shape;
                if (!ds.metadata) ds.metadata = {};
                ds.metadata.label_column = column;
                ds.metadata.label_distribution = r.label_distribution;
            }

            const badges = document.getElementById('dsViewBadges');
            if (badges && ds) {
                badges.innerHTML = `
                    <span class="ds-badge type">${this.fmtType(ds.dataset_type)}</span>
                    <span class="ds-badge samples">${ds.num_samples.toLocaleString()} samples</span>
                    ${ds.num_classes > 0 ? `<span class="ds-badge classes">${ds.num_classes} classes</span>` : ''}
                    <span class="ds-badge shape">[${ds.input_shape.join(', ')}]</span>
                `;
            }

            this.renderOverview(ds);
        } else {
            status.textContent = `✗ ${r.errors.join(', ')}`;
            status.className = 'ds-target-status error';
        }
    } catch (e) {
        status.textContent = `✗ Failed: ${e.message}`;
        status.className = 'ds-target-status error';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Apply';
    }
};

DatasetManagerUI.prototype.showPreview = async function () {
    if (!this.selectedId) return;
    const c = document.getElementById('dsPreviewContent');
    c.innerHTML = '<div class="ds-loading">Loading preview...</div>';
    try {
        const res = await fetch(`${DS_API}/datasets/${this.selectedId}/preview?limit=30`);
        const r = await res.json();
        if (!r.valid) { c.innerHTML = `<div class="ds-error">${r.errors.join(', ')}</div>`; return; }
        const ds = this.datasets.find(d => d.id === this.selectedId);
        if (ds && ds.dataset_type === 'tabular_csv') this.renderCsvPreview(r, c);
        else if (ds && ds.dataset_type.includes('image')) this.renderImgPreview(r, c);
    } catch (e) { c.innerHTML = `<div class="ds-error">Preview failed: ${e.message}</div>`; }
};

DatasetManagerUI.prototype.renderCsvPreview = function (data, c) {
    if (!data.headers) { c.innerHTML = '<div class="ds-error">No data</div>'; return; }
    let h = `<div class="preview-header"><span>Showing ${data.showing} of ${data.total} rows</span></div><div class="preview-table-wrap"><table class="preview-table"><thead><tr>`;
    data.headers.forEach(x => { h += `<th>${x}</th>`; });
    h += '</tr></thead><tbody>';
    data.rows.forEach(row => { h += '<tr>'; data.headers.forEach(x => { h += `<td>${row[x]||''}</td>`; }); h += '</tr>'; });
    h += '</tbody></table></div>';
    c.innerHTML = h;
};

DatasetManagerUI.prototype.renderImgPreview = function (data, c) {
    if (!data.images) { c.innerHTML = '<div class="ds-error">No images</div>'; return; }
    let h = `<div class="preview-header"><span>Showing ${data.showing} of ${data.total_images} images</span></div><div class="ds-image-grid">`;
    data.images.forEach(img => {
        h += `<div class="ds-image-card">${img.thumbnail ? `<img src="data:image/jpeg;base64,${img.thumbnail}">` : '<div class="ds-image-placeholder">No preview</div>'}`;
        h += `<div class="ds-image-info"><div class="ds-image-name" title="${img.filename}">${img.filename}</div>`;
        if (img.class) h += `<div class="ds-image-class">${img.class}</div>`;
        h += `<div class="ds-image-meta">${this.fmtSize(img.size)}${img.width ? ` · ${img.width}x${img.height}` : ''}</div></div></div>`;
    });
    h += '</div>'; c.innerHTML = h;
};

DatasetManagerUI.prototype.loadFolder = async function () {
    const name = document.getElementById('dsFolderName').value.trim();
    const path = document.getElementById('dsFolderPath').value.trim();
    if (!path) { window.app.showToast('Enter a folder path', 'warning'); return; }
    const btn = document.getElementById('dsLoadFolderBtn');
    btn.disabled = true; btn.textContent = 'Loading...';
    try {
        const fd = new FormData();
        fd.append('source_path', path);
        if (name) fd.append('name', name);
        const res = await fetch(`${DS_API}/datasets/upload`, { method: 'POST', body: fd });
        const r = await res.json();
        if (r.valid) {
            window.app.showToast(`Loaded "${r.dataset.name}"`, 'success');
            document.getElementById('dsFolderName').value = '';
            document.getElementById('dsFolderPath').value = '';
            await this.loadDatasets();
            this.selectDataset(r.dataset.id);
        } else { window.app.showToast(r.errors.join(', '), 'error'); }
    } catch (e) { window.app.showToast('Failed: ' + e.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M2 2.5A1.5 1.5 0 013.5 1h2.146a1.5 1.5 0 011.06.44l1 1A1.5 1.5 0 018.768 3H11.5A1.5 1.5 0 0113 4.5v7a1.5 1.5 0 01-1.5 1.5h-8A1.5 1.5 0 012 11.5v-9z"/></svg> Load'; }
};

DatasetManagerUI.prototype.browseFolder = function () {
    const input = document.createElement('input');
    input.type = 'file'; input.webkitdirectory = true;
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const path = e.target.files[0].webkitRelativePath.split('/')[0];
            document.getElementById('dsFolderPath').value = path;
            if (!document.getElementById('dsFolderName').value) document.getElementById('dsFolderName').value = path;
        }
    });
    input.click();
};

DatasetManagerUI.prototype.handleUploadDrop = function (e) { e.preventDefault(); document.getElementById('dsUploadDropZone').classList.remove('dragover'); if (e.dataTransfer.files[0]) this.uploadFile(e.dataTransfer.files[0]); };
DatasetManagerUI.prototype.handleUploadFile = function (e) { if (e.target.files[0]) this.uploadFile(e.target.files[0]); e.target.value = ''; };

DatasetManagerUI.prototype.uploadFile = async function (file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv', 'zip'].includes(ext)) { window.app.showToast('Only CSV and ZIP supported', 'warning'); return; }
    const name = document.getElementById('dsUploadFileName').value.trim() || file.name.replace(/\.[^.]+$/, '');
    const dz = document.getElementById('dsUploadDropZone');
    dz.innerHTML = '<div class="ds-loading">Uploading...</div>';
    try {
        const fd = new FormData(); fd.append('file', file); fd.append('name', name);
        const res = await fetch(`${DS_API}/datasets/upload`, { method: 'POST', body: fd });
        const r = await res.json();
        if (r.valid) {
            window.app.showToast(`Loaded "${r.dataset.name}"`, 'success');
            document.getElementById('dsUploadFileName').value = '';
            await this.loadDatasets();
            this.selectDataset(r.dataset.id);
        } else { window.app.showToast(r.errors.join(', '), 'error'); }
    } catch (e) { window.app.showToast('Upload failed: ' + e.message, 'error'); }
    finally { dz.innerHTML = '<svg width="48" height="48" viewBox="0 0 32 32" fill="none"><path d="M16 22V10M16 10l-5 5M16 10l5 5" stroke="currentColor" stroke-width="1.5" opacity="0.5"/><path d="M5 22v4a2 2 0 002 2h18a2 2 0 002-2v-4" stroke="currentColor" stroke-width="1.5" opacity="0.5"/></svg><p>Drop CSV or ZIP here</p><p class="ds-upload-sub">or click to browse</p>'; }
};

DatasetManagerUI.prototype.deleteDataset = async function () {
    if (!this.selectedId) return;
    const ds = this.datasets.find(d => d.id === this.selectedId);
    if (!ds) return;
    if (!confirm(`Delete "${ds.name}"?`)) return;
    try {
        const res = await fetch(`${DS_API}/datasets/${this.selectedId}`, { method: 'DELETE' });
        const r = await res.json();
        if (r.valid) {
            window.app.showToast(`Deleted "${ds.name}"`, 'success');
            this.selectedId = null;
            await this.loadDatasets();
            document.getElementById('dsWelcome').style.display = '';
            document.getElementById('dsView').style.display = 'none';
        } else { window.app.showToast(r.errors.join(', '), 'error'); }
    } catch (e) { window.app.showToast('Delete failed: ' + e.message, 'error'); }
};

DatasetManagerUI.prototype.purgeAll = async function () {
    if (!confirm('Purge ALL datasets? This cannot be undone.')) return;
    try {
        const res = await fetch(`${DS_API}/datasets/purge`, { method: 'POST' });
        const r = await res.json();
        if (r.valid) {
            window.app.showToast(r.message, 'success');
            this.selectedId = null;
            await this.loadDatasets();
            document.getElementById('dsWelcome').style.display = '';
            document.getElementById('dsView').style.display = 'none';
        } else { window.app.showToast(r.errors.join(', '), 'error'); }
    } catch (e) { window.app.showToast('Purge failed: ' + e.message, 'error'); }
};

DatasetManagerUI.prototype.fmtType = function (t) { return { image_classification: 'Image Classification', image_folder: 'Image Folder', tabular_csv: 'Tabular (CSV)' }[t] || t; };
DatasetManagerUI.prototype.fmtSize = function (b) { if (!b) return '0 B'; const u = ['B','KB','MB','GB']; let i = 0, s = b; while (s >= 1024 && i < u.length - 1) { s /= 1024; i++; } return s.toFixed(1) + ' ' + u[i]; };
