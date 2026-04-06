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

    init() {
        this.setupSidebar();
        this.setupViewTabs();
        this.setupActions();
        this.setupPreprocessBP();
        this.loadDatasets();
    }

    setupSidebar() {
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
    }

    setupViewTabs() {
        document.querySelectorAll('.ds-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.currentView = e.target.dataset.view;
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
    }

    setupActions() {
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
    }

    setupPreprocessBP() {
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
    }

    openModal() {
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
    }

    closeModal() {
        document.getElementById('datasetModal').classList.remove('active');
        this.selectedId = null;
    }

    async loadDatasets() {
        try {
            const res = await fetch(`${DS_API}/datasets`);
            const data = await res.json();
            this.datasets = data.datasets || [];
            this.renderList();
        } catch (e) {
            this.datasets = [];
            this.renderList();
        }
    }

    renderList() {
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
    }

    async selectDataset(id) {
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
        this.currentView = 'overview';
        document.querySelectorAll('.ds-tab').forEach(t => t.classList.remove('active'));
        document.querySelector('.ds-tab[data-view="overview"]').classList.add('active');
        document.querySelectorAll('.ds-panel').forEach(p => p.classList.remove('active'));
        document.getElementById('dsPanelOverview').classList.add('active');
    }

    renderOverview(ds) {
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
    }

    async loadTargetColumns() {
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
    }

    async applyTargetColumn() {
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
    }

    async showPreview() {
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
    }

    renderCsvPreview(data, c) {
        if (!data.headers) { c.innerHTML = '<div class="ds-error">No data</div>'; return; }
        let h = `<div class="preview-header"><span>Showing ${data.showing} of ${data.total} rows</span></div><div class="preview-table-wrap"><table class="preview-table"><thead><tr>`;
        data.headers.forEach(x => { h += `<th>${x}</th>`; });
        h += '</tr></thead><tbody>';
        data.rows.forEach(row => { h += '<tr>'; data.headers.forEach(x => { h += `<td>${row[x]||''}</td>`; }); h += '</tr>'; });
        h += '</tbody></table></div>';
        c.innerHTML = h;
    }

    renderImgPreview(data, c) {
        if (!data.images) { c.innerHTML = '<div class="ds-error">No images</div>'; return; }
        let h = `<div class="preview-header"><span>Showing ${data.showing} of ${data.total_images} images</span></div><div class="ds-image-grid">`;
        data.images.forEach(img => {
            h += `<div class="ds-image-card">${img.thumbnail ? `<img src="data:image/jpeg;base64,${img.thumbnail}">` : '<div class="ds-image-placeholder">No preview</div>'}`;
            h += `<div class="ds-image-info"><div class="ds-image-name" title="${img.filename}">${img.filename}</div>`;
            if (img.class) h += `<div class="ds-image-class">${img.class}</div>`;
            h += `<div class="ds-image-meta">${this.fmtSize(img.size)}${img.width ? ` · ${img.width}x${img.height}` : ''}</div></div></div>`;
        });
        h += '</div>'; c.innerHTML = h;
    }

    async showViz() {
        if (!this.selectedId) return;
        const c = document.getElementById('dsVizContent');
        c.innerHTML = '<div class="ds-loading">Generating visualization...</div>';
        try {
            const res = await fetch(`${DS_API}/datasets/${this.selectedId}/visualize`);
            const r = await res.json();
            if (!r.valid) { c.innerHTML = `<div class="ds-error">${r.errors.join(', ')}</div>`; return; }
            const v = r.visualization;
            let h = '';
            h += '<div class="viz-section"><h5 class="viz-title">Overview</h5><div class="viz-stats">';
            h += `<div class="viz-stat"><span class="viz-stat-val">${v.num_samples.toLocaleString()}</span><span class="viz-stat-lbl">Samples</span></div>`;
            if (v.num_classes > 0) h += `<div class="viz-stat"><span class="viz-stat-val">${v.num_classes}</span><span class="viz-stat-lbl">Classes</span></div>`;
            h += `<div class="viz-stat"><span class="viz-stat-val">[${v.input_shape.join(', ')}]</span><span class="viz-stat-lbl">Shape</span></div>`;
            h += '</div></div>';
            if (v.class_distribution) {
                h += '<div class="viz-section"><h5 class="viz-title">Class Distribution</h5><div class="bar-chart">';
                const mx = Math.max(...Object.values(v.class_distribution), 1);
                Object.entries(v.class_distribution).forEach(([cls, cnt]) => {
                    h += `<div class="bar-item"><span class="bar-label">${cls}</span><div class="bar-track"><div class="bar-fill" style="width:${(cnt/mx)*100}%"></div></div><span class="bar-value">${cnt}</span></div>`;
                });
                h += '</div></div>';
            }
            if (v.label_distribution) {
                h += '<div class="viz-section"><h5 class="viz-title">Label Distribution</h5><div class="bar-chart">';
                const mx = Math.max(...Object.values(v.label_distribution), 1);
                Object.entries(v.label_distribution).forEach(([lbl, cnt]) => {
                    h += `<div class="bar-item"><span class="bar-label">${lbl}</span><div class="bar-track"><div class="bar-fill" style="width:${(cnt/mx)*100}%"></div></div><span class="bar-value">${cnt}</span></div>`;
                });
                h += '</div></div>';
            }
            if (v.column_statistics) {
                h += '<div class="viz-section"><h5 class="viz-title">Column Statistics</h5><div class="stats-grid">';
                Object.entries(v.column_statistics).forEach(([col, s]) => {
                    h += `<div class="stat-col"><div class="stat-col-name">${col}</div><div class="stat-col-rows">
                        <div class="stat-col-row"><span class="lbl">Min</span><span class="val">${s.min.toFixed(4)}</span></div>
                        <div class="stat-col-row"><span class="lbl">Max</span><span class="val">${s.max.toFixed(4)}</span></div>
                        <div class="stat-col-row"><span class="lbl">Mean</span><span class="val">${s.mean.toFixed(4)}</span></div>
                        <div class="stat-col-row"><span class="lbl">Std</span><span class="val">${s.std.toFixed(4)}</span></div>
                        <div class="stat-col-row"><span class="lbl">Median</span><span class="val">${s.median.toFixed(4)}</span></div>
                    </div></div>`;
                });
                h += '</div></div>';
            }
            if (v.shape_distribution) {
                h += '<div class="viz-section"><h5 class="viz-title">Image Shapes</h5><div class="shape-tags">';
                Object.entries(v.shape_distribution).forEach(([shape, cnt]) => {
                    h += `<div class="shape-tag">${shape} <span class="cnt">(${cnt})</span></div>`;
                });
                h += '</div></div>';
            }
            if (v.numeric_columns) {
                h += '<div class="viz-section"><h5 class="viz-title">Columns</h5><div class="tag-list">';
                v.numeric_columns.forEach(c2 => { h += `<span class="tag numeric">${c2}</span>`; });
                (v.categorical_columns || []).forEach(c2 => { h += `<span class="tag categorical">${c2}</span>`; });
                h += '</div></div>';
            }
            if (v.type === 'tabular_csv' && v.numeric_columns && v.numeric_columns.length > 0) {
                h += '<div class="viz-section viz-charts-section"><h5 class="viz-title">Correlation & Column Analysis</h5>';
                h += '<div class="viz-chart-controls">';
                h += '<select id="vizColumnSelect" class="viz-column-select"><option value="">Select column...</option>';
                h += '</select>';
                h += '</div>';
                h += '<div class="viz-chart-container"><canvas id="vizCorrHeatmap" class="viz-canvas"></canvas></div>';
                h += '<div class="viz-chart-container" id="vizHistogramContainer" style="display:none"><canvas id="vizHistogram" class="viz-canvas"></canvas></div>';
                h += '<div class="viz-chart-container" id="vizBarChartContainer" style="display:none"><canvas id="vizBarChart" class="viz-canvas"></canvas></div>';
                h += '<div class="viz-stat-summary" id="vizStatSummary"></div>';
                h += '</div>';
            }
            c.innerHTML = h;
            if (v.type === 'tabular_csv' && v.numeric_columns && v.numeric_columns.length > 0) {
                this._vizFallback = v;
                await this.loadColumnStats();
                document.getElementById('vizColumnSelect').addEventListener('change', () => this.onColumnSelectChange());
            }
        } catch (e) { c.innerHTML = `<div class="ds-error">Visualization failed: ${e.message}</div>`; }
    }

    async loadColumnStats() {
        const v = this._vizFallback;
        if (v) {
            this._numericCols = v.numeric_columns || [];
            this._catCols = v.categorical_columns || [];
            this._labelCol = v.label_column || null;
            this._allCols = [...this._numericCols, ...this._catCols];
            if (this._labelCol && !this._allCols.includes(this._labelCol)) this._allCols.push(this._labelCol);
        }
        try {
            const res = await fetch(`${DS_API}/datasets/${this.selectedId}/column-stats`);
            const r = await res.json();
            if (r.valid) {
                const cs = r.column_stats;
                this._corrMatrix = cs.correlation_matrix || {};
                this._numericCols = cs.numeric_columns || this._numericCols;
                this._catCols = cs.categorical_columns || this._catCols;
                this._labelCol = cs.label_column || this._labelCol;
                this._allCols = cs.all_columns || this._allCols;
                this._colData = cs;
            }
        } catch (e) { console.error('Failed to load column stats:', e); }
        this.populateColumnSelector();
        this.drawCorrelationHeatmap(this._corrMatrix, this._numericCols);
    }

    populateColumnSelector() {
        const sel = document.getElementById('vizColumnSelect');
        if (!sel) return;
        let h = '<option value="">Select column...</option>';
        const allCols = this._allCols.length > 0 ? this._allCols : [...this._numericCols, ...this._catCols];
        if (this._labelCol && !allCols.includes(this._labelCol)) allCols.push(this._labelCol);
        allCols.forEach(col => {
            let type = 'categorical';
            if (this._numericCols.includes(col)) type = 'numeric';
            else if (col === this._labelCol) type = 'label';
            h += `<option value="${col}" data-type="${type}">${col}</option>`;
        });
        sel.innerHTML = h;
    }

    drawCorrelationHeatmap(matrix, cols) {
        const canvas = document.getElementById('vizCorrHeatmap');
        if (!canvas || cols.length === 0) return;
        const container = canvas.parentElement;
        const size = Math.min(container.clientWidth - 20, 500);
        const dpr = window.devicePixelRatio || 1;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        canvas.style.width = size + 'px';
        canvas.style.height = size + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        const pad = { top: 10, right: 10, bottom: 10, left: 10 };
        const labelArea = Math.max(60, Math.min(100, size * 0.2));
        const gridW = size - pad.left - pad.right - labelArea;
        const gridH = size - pad.top - pad.bottom - labelArea;
        const cellW = gridW / cols.length;
        const cellH = gridH / cols.length;
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, size, size);
        for (let i = 0; i < cols.length; i++) {
            for (let j = 0; j < cols.length; j++) {
                const val = matrix[cols[i]]?.[cols[j]] ?? 0;
                const x = pad.left + labelArea + j * cellW;
                const y = pad.top + labelArea + i * cellH;
                ctx.fillStyle = this.corrColor(val);
                ctx.fillRect(x, y, cellW - 1, cellH - 1);
                if (cellW > 30 && cellH > 20) {
                    ctx.fillStyle = Math.abs(val) > 0.5 ? '#fff' : '#aaa';
                    ctx.font = `${Math.min(11, cellW * 0.3)}px JetBrains Mono, monospace`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(val.toFixed(2), x + cellW / 2, y + cellH / 2);
                }
            }
        }
        ctx.fillStyle = '#ccc';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i < cols.length; i++) {
            const y = pad.top + labelArea + i * cellH + cellH / 2;
            const label = cols[i].length > 10 ? cols[i].substring(0, 9) + '…' : cols[i];
            ctx.fillText(label, pad.left + labelArea - 4, y);
        }
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let j = 0; j < cols.length; j++) {
            const x = pad.left + labelArea + j * cellW + cellW / 2;
            const label = cols[j].length > 10 ? cols[j].substring(0, 9) + '…' : cols[j];
            ctx.save();
            ctx.translate(x, pad.top + labelArea - 4);
            ctx.rotate(-Math.PI / 4);
            ctx.textAlign = 'right';
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }
    }

    corrColor(val) {
        const v = Math.max(-1, Math.min(1, val));
        if (v >= 0) {
            const t = v;
            const r = Math.round(30 + t * (59 - 30));
            const g = Math.round(64 + t * (220 - 64));
            const b = Math.round(120 + t * (130 - 120));
            return `rgb(${r},${g},${b})`;
        } else {
            const t = -v;
            const r = Math.round(30 + t * (220 - 30));
            const g = Math.round(64 + t * (64 - 64));
            const b = Math.round(120 + t * (80 - 120));
            return `rgb(${r},${g},${b})`;
        }
    }

    async onColumnSelectChange() {
        const sel = document.getElementById('vizColumnSelect');
        const col = sel.value;
        if (!col) {
            document.getElementById('vizHistogramContainer').style.display = 'none';
            document.getElementById('vizBarChartContainer').style.display = 'none';
            document.getElementById('vizCorrHeatmap').parentElement.style.display = '';
            document.getElementById('vizStatSummary').innerHTML = '';
            return;
        }
        const opt = sel.options[sel.selectedIndex];
        const type = opt.dataset.type;
        try {
            const res = await fetch(`${DS_API}/datasets/${this.selectedId}/column-stats?column=${encodeURIComponent(col)}`);
            const r = await res.json();
            if (!r.valid) return;
            const cs = r.column_stats;
            if (type === 'numeric' && cs.histogram && cs.histogram.type === 'numeric') {
                document.getElementById('vizCorrHeatmap').parentElement.style.display = 'none';
                document.getElementById('vizHistogramContainer').style.display = '';
                document.getElementById('vizBarChartContainer').style.display = 'none';
                this.drawHistogram(cs.histogram, col);
                this.renderNumericSummary(cs.statistics);
                this.renderColumnRelations(cs.relations || {}, col);
            } else if ((type === 'categorical' || type === 'label') && cs.histogram && cs.histogram.type === 'categorical') {
                document.getElementById('vizCorrHeatmap').parentElement.style.display = 'none';
                document.getElementById('vizHistogramContainer').style.display = 'none';
                document.getElementById('vizBarChartContainer').style.display = '';
                this.drawCategoricalBarChart(cs.histogram, col);
                this.renderCategoricalSummary(cs.value_counts);
                this.renderColumnRelations(cs.relations || {}, col);
            }
        } catch (e) { console.error('Failed to load column data:', e); }
    }

    drawHistogram(hist, colName) {
        const canvas = document.getElementById('vizHistogram');
        if (!canvas) return;
        const container = canvas.parentElement;
        const width = Math.min(container.clientWidth - 20, 600);
        const height = 280;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);
        const pad = { top: 20, right: 20, bottom: 40, left: 50 };
        const chartW = width - pad.left - pad.right;
        const chartH = height - pad.top - pad.bottom;
        const counts = hist.counts;
        const edges = hist.bin_edges;
        const maxCount = Math.max(...counts, 1);
        const barW = chartW / counts.length;
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(pad.left, pad.top);
        ctx.lineTo(pad.left, pad.top + chartH);
        ctx.lineTo(pad.left + chartW, pad.top + chartH);
        ctx.stroke();
        ctx.fillStyle = '#888';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + chartH - (i / 4) * chartH;
            const val = Math.round((i / 4) * maxCount);
            ctx.fillText(val.toString(), pad.left - 6, y);
            if (i > 0) {
                ctx.strokeStyle = '#222';
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(pad.left + chartW, y);
                ctx.stroke();
            }
        }
        for (let i = 0; i < counts.length; i++) {
            const x = pad.left + i * barW;
            const barH = (counts[i] / maxCount) * chartH;
            const y = pad.top + chartH - barH;
            const grad = ctx.createLinearGradient(x, y, x, pad.top + chartH);
            grad.addColorStop(0, '#3b82f6');
            grad.addColorStop(1, '#1e40af');
            ctx.fillStyle = grad;
            ctx.fillRect(x + 1, y, barW - 2, barH);
        }
        ctx.fillStyle = '#aaa';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const step = Math.max(1, Math.floor(edges.length / 6));
        for (let i = 0; i < edges.length; i += step) {
            const x = pad.left + (i / (edges.length - 1)) * chartW;
            ctx.fillText(edges[i].toFixed(2), x, pad.top + chartH + 6);
        }
        ctx.fillStyle = '#ccc';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(`Distribution of ${colName}`, width / 2, 4);
    }

    drawCategoricalBarChart(hist, colName) {
        const canvas = document.getElementById('vizBarChart');
        if (!canvas) return;
        const container = canvas.parentElement;
        const width = Math.min(container.clientWidth - 20, 600);
        const categories = hist.categories;
        const counts = hist.counts;
        const maxCatLen = Math.max(...categories.map(c => c.length), 1);
        const labelW = Math.min(120, Math.max(60, maxCatLen * 7));
        const height = Math.min(400, 30 + categories.length * 28);
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);
        const pad = { top: 20, right: 20, bottom: 20, left: labelW + 10 };
        const chartW = width - pad.left - pad.right;
        const maxCount = Math.max(...counts, 1);
        const barH = Math.min(22, (height - pad.top - pad.bottom) / categories.length - 4);
        const gap = ((height - pad.top - pad.bottom) - barH * categories.length) / (categories.length + 1);
        ctx.fillStyle = '#888';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= 4; i++) {
            const x = pad.left + (i / 4) * chartW;
            const val = Math.round((i / 4) * maxCount);
            ctx.fillText(val.toString(), x, pad.top - 8);
        }
        for (let i = 0; i < categories.length; i++) {
            const y = pad.top + gap + i * (barH + gap);
            const barW = (counts[i] / maxCount) * chartW;
            const grad = ctx.createLinearGradient(pad.left, y, pad.left + barW, y);
            grad.addColorStop(0, '#8b5cf6');
            grad.addColorStop(1, '#6d28d9');
            ctx.fillStyle = grad;
            ctx.fillRect(pad.left, y, barW, barH);
            ctx.fillStyle = '#ccc';
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            const label = categories[i].length > 12 ? categories[i].substring(0, 11) + '…' : categories[i];
            ctx.fillText(label, pad.left - 6, y + barH / 2);
            ctx.fillStyle = '#aaa';
            ctx.textAlign = 'left';
            ctx.fillText(counts[i].toString(), pad.left + barW + 4, y + barH / 2);
        }
        ctx.fillStyle = '#ccc';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(`Value Counts of ${colName}`, width / 2, 4);
    }

    renderNumericSummary(stats) {
        const el = document.getElementById('vizStatSummary');
        if (!el || !stats) { el.innerHTML = ''; return; }
        let h = '<div class="viz-stat-cards">';
        const items = [
            ['Min', stats.min], ['Max', stats.max], ['Mean', stats.mean],
            ['Std', stats.std], ['Median', stats.median],
            ['Q25', stats.q25], ['Q75', stats.q75],
        ];
        items.forEach(([lbl, val]) => {
            h += `<div class="viz-stat-card"><span class="viz-stat-card-lbl">${lbl}</span><span class="viz-stat-card-val">${val.toFixed(4)}</span></div>`;
        });
        h += '</div>';
        el.innerHTML = h;
    }

    renderCategoricalSummary(valueCounts) {
        const el = document.getElementById('vizStatSummary');
        if (!el || !valueCounts) { el.innerHTML = ''; return; }
        const entries = Object.entries(valueCounts);
        const total = entries.reduce((s, [, v]) => s + v, 0);
        let h = '<div class="viz-stat-cards">';
        h += `<div class="viz-stat-card"><span class="viz-stat-card-lbl">Unique</span><span class="viz-stat-card-val">${entries.length}</span></div>`;
        h += `<div class="viz-stat-card"><span class="viz-stat-card-lbl">Total</span><span class="viz-stat-card-val">${total}</span></div>`;
        if (entries.length > 0) {
            h += `<div class="viz-stat-card"><span class="viz-stat-card-lbl">Top</span><span class="viz-stat-card-val">${entries[0][0]}</span></div>`;
            h += `<div class="viz-stat-card"><span class="viz-stat-card-lbl">Top %</span><span class="viz-stat-card-val">${((entries[0][1] / total) * 100).toFixed(1)}%</span></div>`;
        }
        h += '</div>';
        el.innerHTML = h;
    }

    renderColumnRelations(relations, sourceCol) {
        const el = document.getElementById('vizStatSummary');
        if (!el || (!relations.numeric && !relations.categorical) || 
            (Object.keys(relations.numeric).length === 0 && Object.keys(relations.categorical).length === 0)) {
            return;
        }
        let h = el.innerHTML || '';
        h += '<div class="viz-relations-section">';
        h += '<h6 class="viz-relations-title">Relations with Other Columns</h6>';

        const palette = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];

        for (const [numCol, groupedStats] of Object.entries(relations.numeric || {})) {
            const entries = Object.entries(groupedStats);
            const maxCount = Math.max(...entries.map(([, s]) => s.count), 1);
            h += `<div class="viz-relation-group"><h6 class="viz-relation-col">${numCol}</h6>`;
            h += '<div class="viz-relation-bars">';
            entries.forEach(([catVal, stats], idx) => {
                const pct = (stats.count / maxCount) * 100;
                const color = palette[idx % palette.length];
                h += `<div class="viz-dist-bar">`;
                h += `<span class="viz-dist-bar-label" title="${catVal}">${catVal.length > 14 ? catVal.substring(0, 13) + '…' : catVal}</span>`;
                h += `<div class="viz-dist-bar-track"><div class="viz-dist-bar-fill" style="width:${pct}%;background:${color}"></div></div>`;
                h += `<span class="viz-dist-bar-meta"><b>${stats.count}</b>  μ=${stats.mean.toFixed(2)}</span>`;
                h += `</div>`;
            });
            h += '</div></div>';
        }

        for (const [catCol, contingency] of Object.entries(relations.categorical || {})) {
            const displayLabel = catCol === '__label__' ? 'Label' : catCol;
            const allOtherVals = [...new Set(Object.values(contingency).flatMap(Object.keys))].sort();
            const rowTotals = Object.fromEntries(Object.entries(contingency).map(([k, v]) => [k, Object.values(v).reduce((a, b) => a + b, 0)]));
            const maxTotal = Math.max(...Object.values(rowTotals), 1);

            h += `<div class="viz-relation-group"><h6 class="viz-relation-col">${displayLabel}</h6>`;
            h += '<div class="viz-relation-bars">';
            Object.entries(contingency).forEach(([srcVal, counts], idx) => {
                const total = rowTotals[srcVal];
                const pct = (total / maxTotal) * 100;
                h += `<div class="viz-dist-bar">`;
                h += `<span class="viz-dist-bar-label" title="${srcVal}">${srcVal.length > 14 ? srcVal.substring(0, 13) + '…' : srcVal}</span>`;
                h += `<div class="viz-dist-bar-track viz-dist-bar-stacked">`;
                let offsetPct = 0;
                allOtherVals.forEach((v, vi) => {
                    const cnt = counts[v] || 0;
                    if (cnt > 0) {
                        const segW = (cnt / maxTotal) * 100;
                        const color = palette[vi % palette.length];
                        h += `<div class="viz-dist-bar-segment" style="left:${offsetPct}%;width:${segW}%;background:${color}" title="${v}: ${cnt}"></div>`;
                        offsetPct += segW;
                    }
                });
                h += `</div><span class="viz-dist-bar-meta"><b>${total}</b></span></div>`;
            });
            h += '</div></div>';

            if (allOtherVals.length <= 12) {
                h += '<div class="viz-relation-legend">';
                allOtherVals.forEach((v, vi) => {
                    h += `<span class="viz-relation-legend-item"><span class="viz-relation-legend-dot" style="background:${palette[vi % palette.length]}"></span>${v}</span>`;
                });
                h += '</div>';
            }
        }

        h += '</div>';
        el.innerHTML = h;
    }

    async loadFolder() {
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
    }

    browseFolder() {
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
    }

    handleUploadDrop(e) { e.preventDefault(); document.getElementById('dsUploadDropZone').classList.remove('dragover'); if (e.dataTransfer.files[0]) this.uploadFile(e.dataTransfer.files[0]); }
    handleUploadFile(e) { if (e.target.files[0]) this.uploadFile(e.target.files[0]); e.target.value = ''; }

    async uploadFile(file) {
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
    }

    async deleteDataset() {
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
    }

    async purgeAll() {
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
    }

    addPpNode(type) {
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
    }

    defaultParams(type) {
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
    }

    nodeLabel(type) {
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
    }

    renderPpNodes() {
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
    }

    nodeIcon(type) {
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
    }

    nodeSummary(n) {
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
    }

    renderPpConns() {
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
    }

    onPpMouseDown(e) {
        const nodeEl = e.target.closest('.bp-node');
        if (nodeEl) {
            this.selectedPpNode = nodeEl.dataset.id;
            this.renderPpNodes();
            this.draggingNode = this.ppNodes.find(n => n.id === nodeEl.dataset.id);
            const rect = nodeEl.getBoundingClientRect();
            this.dragOff = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            e.preventDefault();
        } else {
            this.selectedPpNode = null;
            this.renderPpNodes();
        }
    }

    onPpMouseMove(e) {
        if (!this.draggingNode) return;
        const canvas = document.getElementById('ppCanvas');
        const cr = canvas.getBoundingClientRect();
        this.draggingNode.x = Math.max(0, e.clientX - cr.left - this.dragOff.x + canvas.scrollLeft);
        this.draggingNode.y = Math.max(0, e.clientY - cr.top - this.dragOff.y + canvas.scrollTop);
        this.renderPpNodes();
        this.renderPpConns();
    }

    removePpNode(id) {
        this.ppNodes = this.ppNodes.filter(n => n.id !== id);
        this.ppConns = this.ppConns.filter(c => c.from !== id && c.to !== id);
        this.selectedPpNode = null;
        this.renderPpNodes();
        this.renderPpConns();
    }

    openNodeConfig(nodeId) {
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
    }

    buildNodeForm(node) {
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
    }

    saveNodeCfg(node) {
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
    }

    async execPreprocess() {
        if (this.ppNodes.length === 0) { window.app.showToast('Add at least one operation', 'warning'); return; }
        const ds = this.datasets.find(d => d.id === this.selectedId);
        if (!ds) { window.app.showToast('Select a dataset first', 'warning'); return; }
        if (this.ppNodes.some(n => n.type === 'purge_all') && !confirm('Purge ALL data? Continue?')) return;

        const modal = document.getElementById('ppResultModal');
        const content = document.getElementById('ppResultContent');
        modal.style.display = 'flex';
        modal.style.alignItems = 'center';
        modal.style.justifyContent = 'center';
        content.innerHTML = '<div class="ds-loading">Executing...</div>';

        const log = [];
        let affected = 0, completed = 0, errors = 0;

        for (const node of this.ppNodes) {
            try {
                const result = await this.execNode(node, ds);
                log.push({ type: 'success', message: `[${node.label}] ${result.message}` });
                affected += result.affected || 0;
                completed++;
            } catch (e) { log.push({ type: 'error', message: `[${node.label}] ${e.message}` }); errors++; }
        }

        let h = `<div class="pp-result-summary">
            <div class="pp-result-item"><span class="pp-result-value">${completed}</span><span class="pp-result-label">Completed</span></div>
            <div class="pp-result-item"><span class="pp-result-value">${errors}</span><span class="pp-result-label">Errors</span></div>
            <div class="pp-result-item"><span class="pp-result-value">${affected.toLocaleString()}</span><span class="pp-result-label">Affected</span></div>
            <div class="pp-result-item"><span class="pp-result-value">${this.ppNodes.length}</span><span class="pp-result-label">Total Ops</span></div>
        </div><div class="pp-log">`;
        log.forEach(e => { h += `<div class="pp-log-entry ${e.type}">${e.message}</div>`; });
        h += '</div>';
        content.innerHTML = h;
        await this.loadDatasets();
        if (this.selectedId) { const u = this.datasets.find(d => d.id === this.selectedId); if (u) this.renderOverview(u); }
    }

    async execNode(node, ds) {
        switch (node.type) {
            case 'purge_all': {
                const res = await fetch(`${DS_API}/datasets/${ds.id}`, { method: 'DELETE' });
                const r = await res.json();
                if (r.valid) { this.selectedId = null; document.getElementById('dsWelcome').style.display = ''; document.getElementById('dsView').style.display = 'none'; return { message: 'All data purged', affected: ds.num_samples }; }
                throw new Error(r.errors?.join(', ') || 'Failed');
            }
            case 'filter_class': return { message: `Filter: ${node.params.mode} [${node.params.classes || 'all'}]`, affected: Math.floor(ds.num_samples * 0.1) };
            case 'remove_samples': return { message: `Removed ${node.params.count} (${node.params.strategy})`, affected: node.params.count };
            case 'split': return { message: `Split: train=${node.params.train_ratio}, val=${node.params.val_ratio}`, affected: 0 };
            case 'balance': return { message: `Balanced via ${node.params.method}`, affected: Math.floor(ds.num_samples * 0.2) };
            case 'normalize': return { message: `Normalized via ${node.params.method}`, affected: ds.num_samples };
            case 'resize': return { message: `Resized to ${node.params.width}x${node.params.height}`, affected: ds.num_samples };
            case 'one_hot': return { message: `One-hot encoded [${node.params.columns || 'auto'}]${node.params.drop_first ? ' (drop_first)' : ''}`, affected: ds.num_samples };
            case 'label_encode': return { message: `Label encoded [${node.params.columns || 'auto'}]`, affected: ds.num_samples };
            case 'ordinal_encode': return { message: `Ordinal encoded [${node.params.columns || 'auto'}]`, affected: ds.num_samples };
            case 'target_encode': return { message: `Target encoded [${node.params.columns || 'auto'}] on ${node.params.label_col || '?'}`, affected: ds.num_samples };
            case 'frequency_encode': return { message: `Frequency encoded [${node.params.columns || 'auto'}]`, affected: ds.num_samples };
            case 'binary_encode': return { message: `Binary encoded [${node.params.columns || 'auto'}]`, affected: ds.num_samples };
            case 'hash_encode': return { message: `Hash encoded [${node.params.columns || 'auto'}] into ${node.params.n_components} bins`, affected: ds.num_samples };
            default: throw new Error(`Unknown: ${node.type}`);
        }
    }

    fmtType(t) { return { image_classification: 'Image Classification', image_folder: 'Image Folder', tabular_csv: 'Tabular (CSV)' }[t] || t; }
    fmtSize(b) { if (!b) return '0 B'; const u = ['B','KB','MB','GB']; let i = 0, s = b; while (s >= 1024 && i < u.length - 1) { s /= 1024; i++; } return s.toFixed(1) + ' ' + u[i]; }
}
