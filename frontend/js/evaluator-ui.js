const EVAL_API = 'http://localhost:8000';

class EvaluatorUI {
    constructor() {
        this.modal = document.getElementById('evalModal');
        this.weightSelect = document.getElementById('evalWeightSelect');
        this.weightRefreshBtn = document.getElementById('evalWeightRefreshBtn');
        this.typeBadge = document.getElementById('evalModelTypeBadge');
        this.tabs = document.querySelectorAll('.eval-tab');
        this.tabContents = document.querySelectorAll('.eval-tab-content');
        this.panelImage = document.getElementById('evalPanelImage');
        this.panelTabularCsv = document.getElementById('evalPanelTabularCsv');
        this.panelTabularSingle = document.getElementById('evalPanelTabularSingle');
        this.imageDropZone = document.getElementById('evalImageDropZone');
        this.imageInput = document.getElementById('evalImageInput');
        this.imageStrip = document.getElementById('evalImageStrip');
        this.imageResults = document.getElementById('evalImageResults');
        this.imagePredGrid = document.getElementById('evalImagePredGrid');
        this.topKSlider = document.getElementById('evalTopK');
        this.topKValue = document.getElementById('evalTopKValue');
        this.imageRunBtn = document.getElementById('evalImageRunBtn');
        this.csvDropZone = document.getElementById('evalCsvDropZone');
        this.csvInput = document.getElementById('evalCsvInput');
        this.csvRunBtn = document.getElementById('evalCsvRunBtn');
        this.csvDownloadBtn = document.getElementById('evalCsvDownloadBtn');
        this.csvResults = document.getElementById('evalCsvResults');
        this.csvPreviewTable = document.getElementById('evalCsvPreviewTable');
        this.singleForm = document.getElementById('evalSingleRowForm');
        this.singleRunBtn = document.getElementById('evalSingleRunBtn');
        this.singleResults = document.getElementById('evalSingleResults');
        this.singlePredResult = document.getElementById('evalSinglePredResult');
        this.loading = document.getElementById('evalLoading');

        this.imageFiles = [];
        this.csvFile = null;
        this.modelType = null;
        this.inputShape = null;
        this.numClasses = null;
        this.blueprint = null;
        this.csvResultBlob = null;
    }

    init() {
        this.topKSlider.addEventListener('input', () => {
            this.topKValue.textContent = this.topKSlider.value;
        });

        this.tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const target = e.currentTarget.dataset.tab;
                this.tabs.forEach(t => t.classList.remove('active'));
                e.currentTarget.classList.add('active');
                this.tabContents.forEach(tc => tc.classList.remove('active'));
                const panel = document.getElementById(`evalPanel${target.charAt(0).toUpperCase()}${target.slice(1)}`);
                if (panel) panel.classList.add('active');
            });
        });

        this.imageDropZone.addEventListener('click', () => this.imageInput.click());
        this.imageDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.imageDropZone.classList.add('dragover');
        });
        this.imageDropZone.addEventListener('dragleave', () => {
            this.imageDropZone.classList.remove('dragover');
        });
        this.imageDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.imageDropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleImageFiles(e.dataTransfer.files);
            }
        });
        this.imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageFiles(e.target.files);
            }
            e.target.value = '';
        });
        this.imageRunBtn.addEventListener('click', () => this.runImageEvaluation());

        this.csvDropZone.addEventListener('click', () => this.csvInput.click());
        this.csvDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.csvDropZone.classList.add('dragover');
        });
        this.csvDropZone.addEventListener('dragleave', () => {
            this.csvDropZone.classList.remove('dragover');
        });
        this.csvDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.csvDropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleCsvFile(e.dataTransfer.files[0]);
            }
        });
        this.csvInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleCsvFile(e.target.files[0]);
            }
            e.target.value = '';
        });
        this.csvRunBtn.addEventListener('click', () => this.runCsvEvaluation());
        this.csvDownloadBtn.addEventListener('click', () => this.downloadCsvResults());
        this.singleRunBtn.addEventListener('click', () => this.runSingleEvaluation());
        this.weightRefreshBtn.addEventListener('click', () => this.loadWeights());
    }

    open() {
        this.blueprint = window.app.getBlueprint();

        if (!this.blueprint || this.blueprint.layers.length === 0) {
            window.app.showToast('Add layers to the canvas first!', 'warning');
            return;
        }

        this.modal.classList.add('active');
        this.clearResults();
        this.detectModelType(this.blueprint);
        this.loadWeights();
    }

    close() {
        this.modal.classList.remove('active');
        this.imageFiles = [];
        this.csvFile = null;
        this.csvResultBlob = null;
    }

    async detectModelType(blueprint) {
        try {
            const response = await fetch(`${EVAL_API}/evaluate/detect-type`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(blueprint)
            });
            const data = await response.json();

            if (!data.type) {
                this.typeBadge.textContent = 'Unknown';
                this.typeBadge.className = 'eval-type-badge unknown';
                this.showAllTabs();
                this.modelType = null;
                this.inputShape = null;
                this.numClasses = null;
                return;
            }

            this.modelType = data.type || 'unknown';
            this.inputShape = data.input_shape || null;
            this.numClasses = data.num_classes || null;

            this.typeBadge.textContent = this.modelType.charAt(0).toUpperCase() + this.modelType.slice(1);
            this.typeBadge.className = 'eval-type-badge';
            if (this.modelType === 'image') this.typeBadge.classList.add('image');
            else if (this.modelType === 'tabular') this.typeBadge.classList.add('tabular');
            else this.typeBadge.classList.add('unknown');

            this.tabs.forEach(tab => {
                const t = tab.dataset.tab;
                if (this.modelType === 'image') {
                    tab.style.display = t === 'image' ? '' : 'none';
                } else if (this.modelType === 'tabular') {
                    tab.style.display = (t === 'tabular-csv' || t === 'tabular-single') ? '' : 'none';
                } else {
                    tab.style.display = '';
                }
            });

            const firstVisible = Array.from(this.tabs).find(t => t.style.display !== 'none');
            if (firstVisible) {
                this.tabs.forEach(t => t.classList.remove('active'));
                firstVisible.classList.add('active');
                this.tabContents.forEach(tc => tc.classList.remove('active'));
                const panel = document.getElementById(`evalPanel${firstVisible.dataset.tab.charAt(0).toUpperCase()}${firstVisible.dataset.tab.slice(1)}`);
                if (panel) panel.classList.add('active');
            }

            if (this.modelType === 'tabular' && this.inputShape && this.inputShape.features) {
                this.buildSingleRowForm(this.inputShape.features);
            }
        } catch (e) {
            this.typeBadge.textContent = 'Unknown';
            this.typeBadge.className = 'eval-type-badge unknown';
            this.showAllTabs();
            this.modelType = null;
            this.inputShape = null;
            this.numClasses = null;
        }
    }

    showAllTabs() {
        this.tabs.forEach(tab => { tab.style.display = ''; });
        this.tabs.forEach(t => t.classList.remove('active'));
        this.tabContents.forEach(tc => tc.classList.remove('active'));
        const firstTab = this.tabs[0];
        if (firstTab) {
            firstTab.classList.add('active');
            const panel = document.getElementById('evalPanelImage');
            if (panel) panel.classList.add('active');
        }
    }

    async loadWeights() {
        try {
            const response = await fetch(`${EVAL_API}/weights`);
            const data = await response.json();
            const weights = data.weights || [];

            this.weightSelect.innerHTML = '<option value="">No weights</option>';
            weights.forEach(w => {
                const opt = document.createElement('option');
                opt.value = w.filename;
                opt.textContent = `${w.filename} (${w.size_human})`;
                this.weightSelect.appendChild(opt);
            });
        } catch (e) {
            this.weightSelect.innerHTML = '<option value="">No weights</option>';
        }
    }

    handleImageFiles(files) {
        const maxFiles = 50;
        const fileArray = Array.from(files).slice(0, maxFiles);
        this.imageFiles = fileArray;
        this.renderImageThumbnails();
    }

    renderImageThumbnails() {
        this.imageStrip.innerHTML = '';
        if (this.imageFiles.length === 0) {
            this.imageStrip.style.display = 'none';
            return;
        }

        this.imageStrip.style.display = 'flex';
        this.imageFiles.forEach(file => {
            const thumb = document.createElement('div');
            thumb.className = 'eval-image-thumb';
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = file.name;
            const label = document.createElement('span');
            label.className = 'eval-thumb-label';
            label.textContent = file.name;
            thumb.appendChild(img);
            thumb.appendChild(label);
            this.imageStrip.appendChild(thumb);
        });
    }

    async runImageEvaluation() {
        if (this.imageFiles.length === 0) {
            window.app.showToast('Select at least one image first!', 'warning');
            return;
        }

        this.showLoading(true);
        this.imageRunBtn.disabled = true;
        this.imageRunBtn.textContent = 'Evaluating...';

        try {
            const fd = new FormData();
            fd.append('blueprint', JSON.stringify(this.blueprint));
            fd.append('weights_filename', this.weightSelect.value || '');
            fd.append('top_k', this.topKSlider.value);

            this.imageFiles.forEach(file => {
                fd.append('images', file);
            });

            const response = await fetch(`${EVAL_API}/evaluate/image`, {
                method: 'POST',
                body: fd
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            this.renderImagePredictions(data);
            window.app.showToast('Image evaluation complete!', 'success');
        } catch (error) {
            window.app.showToast('Image evaluation failed: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
            this.imageRunBtn.disabled = false;
            this.imageRunBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M11.251.068a.5.5 0 01.227.58L9.677 6.5H13a.5.5 0 01.364.843l-8 8.5a.5.5 0 01-.842-.49L6.323 9.5H3a.5.5 0 01-.364-.843l8-8.5a.5.5 0 01.615-.09z"/>
                </svg>
                Run Evaluation
            `;
        }
    }

    renderImagePredictions(data) {
        this.imageResults.style.display = 'block';
        this.imagePredGrid.innerHTML = '';

        if (!data.predictions || data.predictions.length === 0) {
            this.imagePredGrid.innerHTML = '<p class="eval-no-data">No predictions returned.</p>';
            return;
        }

        data.predictions.forEach((preds, imgIdx) => {
            const card = document.createElement('div');
            card.className = 'eval-prediction-card';

            const header = document.createElement('div');
            header.className = 'eval-prediction-card-header';
            const imgName = this.imageFiles[imgIdx] ? this.imageFiles[imgIdx].name : `Image ${imgIdx + 1}`;
            header.innerHTML = `<span class="eval-prediction-card-image">${imgName}</span>`;

            const list = document.createElement('div');

            (preds || []).forEach(pred => {
                const item = document.createElement('div');
                item.className = 'eval-prediction-item';

                const label = document.createElement('span');
                label.className = 'eval-prediction-class';
                label.textContent = pred.label !== undefined ? `Class ${pred.label}` : 'Unknown';

                const barTrack = document.createElement('div');
                barTrack.className = 'eval-prediction-bar';

                const barFill = document.createElement('div');
                barFill.className = 'eval-prediction-fill';
                const pct = (pred.confidence * 100).toFixed(1);
                barFill.style.width = `${pct}%`;

                const value = document.createElement('span');
                value.className = 'eval-prediction-value';
                value.textContent = `${pct}%`;

                barTrack.appendChild(barFill);
                item.appendChild(label);
                item.appendChild(barTrack);
                item.appendChild(value);
                list.appendChild(item);
            });

            card.appendChild(header);
            card.appendChild(list);
            this.imagePredGrid.appendChild(card);
        });
    }

    handleCsvFile(file) {
        if (!file.name.endsWith('.csv')) {
            window.app.showToast('Please select a CSV file.', 'warning');
            return;
        }
        this.csvFile = file;
        const label = this.csvDropZone.querySelector('.dataset-drop-text');
        if (label) label.textContent = `Selected: ${file.name}`;
    }

    async runCsvEvaluation() {
        if (!this.csvFile) {
            window.app.showToast('Select a CSV file first!', 'warning');
            return;
        }

        this.showLoading(true);
        this.csvRunBtn.disabled = true;
        this.csvRunBtn.textContent = 'Evaluating...';

        try {
            const fd = new FormData();
            fd.append('blueprint', JSON.stringify(this.blueprint));
            fd.append('weights_filename', this.weightSelect.value || '');
            fd.append('file', this.csvFile);

            const response = await fetch(`${EVAL_API}/evaluate/tabular`, {
                method: 'POST',
                body: fd
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }

            this.csvResultBlob = await response.blob();
            this.csvDownloadBtn.style.display = '';
            this.csvResults.style.display = 'block';

            const text = await this.csvResultBlob.text();
            const lines = text.split('\n').filter(l => l.trim());
            if (lines.length > 0) {
                const headers = lines[0].split(',');
                const rows = lines.slice(1, 11).map(l => l.split(','));

                this.csvPreviewTable.innerHTML = '';
                const table = document.createElement('table');
                table.className = 'eval-csv-table';

                const thead = document.createElement('thead');
                const trHead = document.createElement('tr');
                headers.forEach(h => {
                    const th = document.createElement('th');
                    th.textContent = h.trim();
                    trHead.appendChild(th);
                });
                thead.appendChild(trHead);
                table.appendChild(thead);

                const tbody = document.createElement('tbody');
                rows.forEach(row => {
                    const tr = document.createElement('tr');
                    row.forEach(cell => {
                        const td = document.createElement('td');
                        td.textContent = cell.trim();
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
                table.appendChild(tbody);

                this.csvPreviewTable.appendChild(table);

                if (lines.length > 11) {
                    const note = document.createElement('p');
                    note.className = 'eval-csv-note';
                    note.textContent = `Showing first 10 of ${lines.length - 1} rows. Download full results.`;
                    this.csvPreviewTable.appendChild(note);
                }
            }

            window.app.showToast('CSV evaluation complete!', 'success');
        } catch (error) {
            window.app.showToast('CSV evaluation failed: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
            this.csvRunBtn.disabled = false;
            this.csvRunBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M11.251.068a.5.5 0 01.227.58L9.677 6.5H13a.5.5 0 01.364.843l-8 8.5a.5.5 0 01-.842-.49L6.323 9.5H3a.5.5 0 01-.364-.843l8-8.5a.5.5 0 01.615-.09z"/>
                </svg>
                Run Evaluation
            `;
        }
    }

    downloadCsvResults() {
        if (!this.csvResultBlob) {
            window.app.showToast('No results to download.', 'warning');
            return;
        }
        window.Utils.downloadBlob(this.csvResultBlob, 'evaluation_results.csv');
    }

    buildSingleRowForm(numFeatures) {
        this.singleForm.innerHTML = '';
        for (let i = 0; i < numFeatures; i++) {
            const group = document.createElement('div');
            group.className = 'property-row';

            const label = document.createElement('label');
            label.className = 'property-label';
            label.textContent = `Feature ${i + 1}`;

            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.className = 'property-input';
            input.placeholder = '0';
            input.id = `evalFeature${i}`;

            group.appendChild(label);
            group.appendChild(input);
            this.singleForm.appendChild(group);
        }
    }

    async runSingleEvaluation() {
        const inputs = this.singleForm.querySelectorAll('input');
        const values = Array.from(inputs).map(inp => parseFloat(inp.value) || 0);

        if (values.length === 0) {
            window.app.showToast('No features to evaluate.', 'warning');
            return;
        }

        this.showLoading(true);
        this.singleRunBtn.disabled = true;
        this.singleRunBtn.textContent = 'Evaluating...';

        try {
            const response = await fetch(`${EVAL_API}/evaluate/tabular/single`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    blueprint: this.blueprint,
                    weights_filename: this.weightSelect.value || '',
                    features: values
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            this.singleResults.style.display = 'block';

            if (data.prediction !== undefined) {
                this.singlePredResult.textContent = `Prediction: ${data.prediction}`;
            } else if (data.predictions && data.predictions.length > 0) {
                const preds = data.predictions
                    .map(p => `Class ${p.label}: ${(p.confidence * 100).toFixed(1)}%`)
                    .join('  |  ');
                this.singlePredResult.textContent = preds;
            } else {
                this.singlePredResult.textContent = JSON.stringify(data);
            }

            window.app.showToast('Single evaluation complete!', 'success');
        } catch (error) {
            window.app.showToast('Single evaluation failed: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
            this.singleRunBtn.disabled = false;
            this.singleRunBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M11.251.068a.5.5 0 01.227.58L9.677 6.5H13a.5.5 0 01.364.843l-8 8.5a.5.5 0 01-.842-.49L6.323 9.5H3a.5.5 0 01-.364-.843l8-8.5a.5.5 0 01.615-.09z"/>
                </svg>
                Evaluate
            `;
        }
    }

    clearResults() {
        this.imageResults.style.display = 'none';
        this.imagePredGrid.innerHTML = '';
        this.imageStrip.innerHTML = '';
        this.imageStrip.style.display = 'none';
        this.imageFiles = [];

        this.csvResults.style.display = 'none';
        this.csvPreviewTable.innerHTML = '';
        this.csvDownloadBtn.style.display = 'none';
        this.csvResultBlob = null;
        this.csvFile = null;
        const csvLabel = this.csvDropZone.querySelector('p');
        if (csvLabel) csvLabel.textContent = 'Drop CSV file here';

        this.singleResults.style.display = 'none';
        this.singlePredResult.textContent = '';

        this.showLoading(false);
    }

    showLoading(show) {
        if (this.loading) {
            this.loading.style.display = show ? 'flex' : 'none';
        }
    }
}
