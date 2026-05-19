DatasetManagerUI.prototype.showViz = async function () {
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
};

DatasetManagerUI.prototype.loadColumnStats = async function () {
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
};

DatasetManagerUI.prototype.populateColumnSelector = function () {
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
};

DatasetManagerUI.prototype.drawCorrelationHeatmap = function (matrix, cols) {
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
        const label = cols[i].length > 10 ? cols[i].substring(0, 9) + '\u2026' : cols[i];
        ctx.fillText(label, pad.left + labelArea - 4, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let j = 0; j < cols.length; j++) {
        const x = pad.left + labelArea + j * cellW + cellW / 2;
        const label = cols[j].length > 10 ? cols[j].substring(0, 9) + '\u2026' : cols[j];
        ctx.save();
        ctx.translate(x, pad.top + labelArea - 4);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'right';
        ctx.fillText(label, 0, 0);
        ctx.restore();
    }
};

DatasetManagerUI.prototype.corrColor = function (val) {
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
};

DatasetManagerUI.prototype.onColumnSelectChange = async function () {
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
};

DatasetManagerUI.prototype.drawHistogram = function (hist, colName) {
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
};

DatasetManagerUI.prototype.drawCategoricalBarChart = function (hist, colName) {
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
        const label = categories[i].length > 12 ? categories[i].substring(0, 11) + '\u2026' : categories[i];
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
};

DatasetManagerUI.prototype.renderNumericSummary = function (stats) {
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
};

DatasetManagerUI.prototype.renderCategoricalSummary = function (valueCounts) {
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
};

DatasetManagerUI.prototype.renderColumnRelations = function (relations, sourceCol) {
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
            h += `<span class="viz-dist-bar-label" title="${catVal}">${catVal.length > 14 ? catVal.substring(0, 13) + '\u2026' : catVal}</span>`;
            h += `<div class="viz-dist-bar-track"><div class="viz-dist-bar-fill" style="width:${pct}%;background:${color}"></div></div>`;
            h += `<span class="viz-dist-bar-meta"><b>${stats.count}</b>  \u03BC=${stats.mean.toFixed(2)}</span>`;
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
            h += `<span class="viz-dist-bar-label" title="${srcVal}">${srcVal.length > 14 ? srcVal.substring(0, 13) + '\u2026' : srcVal}</span>`;
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
};
