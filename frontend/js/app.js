class App {
    constructor() {
        this.canvas = new CanvasManager();
        this.nodeManager = new NodeManager(this.canvas);
        this.connectionManager = new ConnectionManager(this.canvas, this.nodeManager);
        this.propertiesPanel = new PropertiesPanel();
        this.codeGenerator = new CodeGenerator();
        this.datasetManager = new DatasetManagerUI();
        this.datasetManager.init();
        
        window.app = this;
        
        this.init();
        this.loadFromLocalStorage();
    }
    
    loadFromLocalStorage() {
        const saved = localStorage.getItem('nnfactory_autosave');
        if (saved) {
            try {
                const blueprint = JSON.parse(saved);
                this.nodeManager.importNodes(blueprint.layers || []);
                this.connectionManager.importConnections(blueprint.connections || []);
                
                if (blueprint.model_name) {
                    const modelNameEl = document.getElementById('modelName');
                    if (modelNameEl) modelNameEl.value = blueprint.model_name;
                }
                if (blueprint.use_jit !== undefined) {
                    const jitEl = document.getElementById('useJit');
                    if (jitEl) jitEl.checked = blueprint.use_jit;
                }
                if (blueprint.use_compile !== undefined) {
                    const compileEl = document.getElementById('useCompile');
                    if (compileEl) compileEl.checked = blueprint.use_compile;
                }
                if (blueprint.device !== undefined) {
                    const deviceEl = document.getElementById('deviceSelect');
                    if (deviceEl) deviceEl.value = blueprint.device;
                }
                
                this.renderConnections();
                // Trigger autosave only after both nodes and connections are fully imported
                this.saveToLocalStorage();
                this.showToast('Session restored!', 'success');
            } catch (error) {
                console.error('Failed to restore session from localStorage:', error);
                this.showToast('Failed to restore previous session.', 'error');
            }
        }
    }

    saveToLocalStorage() {
        try {
            const blueprint = this.getBlueprint();
            localStorage.setItem('nnfactory_autosave', JSON.stringify(blueprint));
        } catch (error) {
            console.error('Failed to save session to localStorage:', error);
        }
    }
    
    init() {
        this.setupEventListeners();
        this.setupCategoryToggles();
        this.renderConnections();
    }
    
    setupEventListeners() {
        document.getElementById('generateBtn').addEventListener('click', () => this.generateCode());
        document.getElementById('validateBtn').addEventListener('click', () => this.validateBlueprint());
        document.getElementById('trainBtn').addEventListener('click', () => this.openTrainModal());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.openEvalModal());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportBlueprint());
        document.getElementById('weightsBtn').addEventListener('click', () => this.openWeightsModal());
        document.getElementById('importBtn').addEventListener('click', () => document.getElementById('fileInput').click());
        document.getElementById('fileInput').addEventListener('change', (e) => this.importBlueprint(e));
        document.getElementById('clearBtn').addEventListener('click', () => this.clearCanvas());
        
        document.getElementById('closeModal').addEventListener('click', () => this.closeCodeModal());
        document.getElementById('closeTrainModal').addEventListener('click', () => this.closeTrainModal());
        document.getElementById('closeEvalModal').addEventListener('click', () => this.closeEvalModal());
        document.getElementById('closeWeightsModal').addEventListener('click', () => this.closeWeightsModal());
        
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                this.closeCodeModal();
                this.closeTrainModal();
                this.closeEvalModal();
                this.closeDatasetModal();
                this.closeWeightsModal();
            });
        });
        
        document.getElementById('copyCode').addEventListener('click', () => this.copyCode());
        document.getElementById('downloadCode').addEventListener('click', () => this.downloadCode());
        document.getElementById('startTrainingBtn').addEventListener('click', () => this.startTraining());
        document.getElementById('stopTrainingBtn').addEventListener('click', () => this.stopTraining());
        document.getElementById('trainAgainBtn').addEventListener('click', () => this.resetTrainModal());
        document.getElementById('exportWeightsBtn').addEventListener('click', () => this.exportWeights());
        document.getElementById('startEvalBtn').addEventListener('click', () => this.startEvaluation());
        document.getElementById('purgeWeightsBtn').addEventListener('click', () => this.purgeWeights());
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeCodeModal();
                this.closeTrainModal();
                this.closeEvalModal();
                this.closeDatasetModal();
                this.closeWeightsModal();
            }
            if ((e.key === 'Delete' || e.key === 'Backspace') && !this.isInputFocused()) {
                if (this.connectionManager.selectedConnection !== null) {
                    e.preventDefault();
                    this.connectionManager.removeConnection(this.connectionManager.selectedConnection);
                    this.connectionManager.selectedConnection = null;
                    this.canvas.render();
                    return;
                }
                if (this.nodeManager.selectedNode) {
                    this.nodeManager.deleteNode(this.nodeManager.selectedNode.id);
                    this.propertiesPanel.hide();
                }
            }
        });
        
 
    }
    
    setupCategoryToggles() {
        document.querySelectorAll('.category-header').forEach(header => {
            header.addEventListener('click', () => {
                const category = header.dataset.category;
                const content = document.querySelector(`.category-content[data-category="${category}"]`);
                
                header.classList.toggle('collapsed');
                content.classList.toggle('collapsed');
            });
        });
    }
    
    renderConnections() {
        requestAnimationFrame(() => {
            this.connectionManager.render();
        });
    }
    
   onNodesChanged() {
        this.renderConnections();
        this.saveToLocalStorage();
        this.nodeManager.updateCounts();
    }
    
    onConnectionsChanged() {
        this.renderConnections();
        this.saveToLocalStorage();
    }
    
    isInputFocused() {
        const active = document.activeElement;
        return active && (active.tagName === 'INPUT' || active.tagName === 'SELECT' || active.tagName === 'TEXTAREA');
    }
    
    getBlueprint() {
        return {
            layers: this.nodeManager.exportNodes(),
            connections: this.connectionManager.getConnectionsArray(),
            model_name: document.getElementById('modelName').value || 'NeuralNetwork',
            use_jit: document.getElementById('useJit').checked,
            use_compile: document.getElementById('useCompile').checked,
            device: document.getElementById('deviceSelect').value || 'cpu'
        };
    }

    async generateCode() {
        const blueprint = this.getBlueprint();
        
        if (blueprint.layers.length === 0) {
            this.showToast('Add some layers to the canvas first!', 'warning');
            return;
        }
        
        const btn = document.getElementById('generateBtn');
        btn.disabled = true;
        btn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" class="spin">
                <path d="M8 3a5 5 0 100 10 5 5 0 000-10z" fill="none" stroke="currentColor" stroke-width="1.5"/>
                <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" stroke="currentColor" stroke-width="1"/>
            </svg>
            Generating...
        `;
        
        try {
            const code = await this.codeGenerator.generateCode(blueprint);
            this.showCodeModal(code);
            this.showToast('Code generated successfully!', 'success');
        } catch (error) {
            this.showToast('Failed to generate code: ' + error.message, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M2 2l6 6-6 6V2zm6 0l6 6-6 6V2z" transform="translate(1,0)"/>
                </svg>
                Generate Code
            `;
        }
    }
    
    async validateBlueprint() {
        const blueprint = this.getBlueprint();
        
        if (blueprint.layers.length === 0) {
            this.showToast('No layers to validate!', 'warning');
            return;
        }
        
        const result = await this.codeGenerator.validateBlueprint(blueprint);
        
        if (result.valid) {
            this.showToast('Blueprint is valid!', 'success');
        } else {
            result.errors.forEach(err => this.showToast(err, 'error'));
        }
    }
    
    showCodeModal(code) {
        const modal = document.getElementById('codeModal');
        const codeEl = document.getElementById('generatedCode');
        codeEl.textContent = code;
        modal.classList.add('active');
    }
    
    closeCodeModal() {
        document.getElementById('codeModal').classList.remove('active');
    }
    
    async copyCode() {
        const code = document.getElementById('generatedCode').textContent;
        try {
            await navigator.clipboard.writeText(code);
            this.showToast('Code copied to clipboard!', 'success');
        } catch {
            const textarea = document.createElement('textarea');
            textarea.value = code;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            this.showToast('Code copied to clipboard!', 'success');
        }
    }
    
    downloadCode() {
        const code = document.getElementById('generatedCode').textContent;
        const modelName = document.getElementById('modelName').value || 'NeuralNetwork';
        const blob = new Blob([code], { type: 'text/python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${modelName.toLowerCase()}.py`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('Code downloaded!', 'success');
    }
    
    exportBlueprint() {
        const blueprint = this.getBlueprint();
        const blob = new Blob([JSON.stringify(blueprint, null, 2)], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${blueprint.model_name.toLowerCase()}.nn`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('Blueprint saved!', 'success');
    }
    
    importBlueprint(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const blueprint = JSON.parse(event.target.result);
                this.clearCanvas();
                this.nodeManager.importNodes(blueprint.layers);
                this.connectionManager.importConnections(blueprint.connections);
                
                if (blueprint.model_name) {
                    document.getElementById('modelName').value = blueprint.model_name;
                }
                if (blueprint.use_jit !== undefined) {
                    document.getElementById('useJit').checked = blueprint.use_jit;
                }
                if (blueprint.use_compile !== undefined) {
                    document.getElementById('useCompile').checked = blueprint.use_compile;
                }
                if (blueprint.device !== undefined) {
                    document.getElementById('deviceSelect').value = blueprint.device;
                }
                
                this.renderConnections();
                this.showToast('Blueprint loaded!', 'success');
            } catch (error) {
                this.showToast('Invalid blueprint file!', 'error');
            }
        };
        reader.readAsText(file);
        e.target.value = '';
    }
    
    clearCanvas() {
        this.nodeManager.clear();
        this.connectionManager.clear();
        this.propertiesPanel.hide();
        this.showToast('Canvas cleared!', 'info');
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 100 16A8 8 0 008 0zm3.78 5.22a.75.75 0 010 1.06l-4 4a.75.75 0 01-1.06 0l-2-2a.75.75 0 111.06-1.06L7 8.44l3.22-3.22a.75.75 0 011.06 0z"/></svg>',
            error: '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 100 16A8 8 0 008 0zm2.53 10.47a.75.75 0 01-1.06 0L8 9l-1.47 1.47a.75.75 0 01-1.06-1.06L6.94 8 5.47 6.53a.75.75 0 011.06-1.06L8 6.94l1.47-1.47a.75.75 0 011.06 1.06L9.06 8l1.47 1.47a.75.75 0 010 1.06z"/></svg>',
            warning: '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 100 16A8 8 0 008 0zm.75 11.25a.75.75 0 01-1.5 0v-2a.75.75 0 011.5 0v2zm0-4a.75.75 0 01-1.5 0v-2a.75.75 0 011.5 0v2z"/></svg>',
            info: '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 100 16A8 8 0 008 0zm.75 11.25a.75.75 0 01-1.5 0v-2a.75.75 0 011.5 0v2zm0-4a.75.75 0 01-1.5 0v-2a.75.75 0 011.5 0v2z"/></svg>'
        };
        
        toast.innerHTML = `${icons[type] || icons.info}<span>${message}</span>`;
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    openTrainModal() {
        if (this.nodeManager.getNodesArray().length === 0) {
            this.showToast('Add layers to the canvas first!', 'warning');
            return;
        }
        this.resetTrainModal();
        this.populateDatasetSelector();
        document.getElementById('trainModal').classList.add('active');
    }

    async populateDatasetSelector() {
        const select = document.getElementById('trainDataset');
        select.value = '';
        try {
            const res = await fetch('http://localhost:8000/datasets');
            const data = await res.json();
            const datasets = data.datasets || [];
            select.innerHTML = '<option value="">Synthetic (random data)</option>';
            datasets.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds.id;
                opt.textContent = `${ds.name} (${ds.num_samples} samples, ${ds.num_classes || '?'} classes)`;
                select.appendChild(opt);
            });
        } catch (e) {}
        this.updateSyntheticFields();
        select.addEventListener('change', () => this.updateSyntheticFields());
    }

    updateSyntheticFields() {
        const datasetId = document.getElementById('trainDataset').value;
        const show = !datasetId;
        document.querySelectorAll('#syntheticFields, #syntheticFields2, #syntheticFields3, #syntheticFields4').forEach(el => {
            el.style.display = show ? '' : 'none';
        });
    }
    
    closeTrainModal() {
        document.getElementById('trainModal').classList.remove('active');
    }
    
    resetTrainModal() {
        document.getElementById('trainConfig').style.display = '';
        document.getElementById('trainProgress').style.display = 'none';
        document.getElementById('trainResults').style.display = 'none';
        document.getElementById('stopTrainingBtn').style.display = 'none';
        document.getElementById('startTrainingBtn').disabled = false;
        this._trainHistory = null;
        this._trainChart = null;
        this._weightsFilename = null;
    }
    
    async startTraining() {
        const blueprint = this.getBlueprint();
        const datasetId = document.getElementById('trainDataset').value;
        
        const baseConfig = {
            blueprint,
            epochs: parseInt(document.getElementById('trainEpochs').value) || 10,
            learning_rate: parseFloat(document.getElementById('trainLR').value) || 0.001,
            batch_size: parseInt(document.getElementById('trainBatchSize').value) || 32,
            optimizer: document.getElementById('trainOptimizer').value,
            loss_function: document.getElementById('trainLoss').value,
            scheduler: document.getElementById('trainScheduler').value,
            weight_decay: parseFloat(document.getElementById('trainWeightDecay').value) || 0.0,
            step_size: 30,
            gamma: 0.1,
            val_ratio: 0.2
        };

        let config;
        let url;

        if (datasetId) {
            config = { ...baseConfig, dataset_id: datasetId };
            url = 'http://localhost:8000/train/dataset';
        } else {
            config = {
                ...baseConfig,
                input_size: [
                    parseInt(document.getElementById('trainInputC').value) || 3,
                    parseInt(document.getElementById('trainInputH').value) || 32,
                    parseInt(document.getElementById('trainInputW').value) || 32
                ],
                num_classes: parseInt(document.getElementById('trainNumClasses').value) || 10,
                num_samples: parseInt(document.getElementById('trainSamples').value) || 1000,
            };
            url = 'http://localhost:8000/train';
        }
        
        document.getElementById('trainConfig').style.display = 'none';
        document.getElementById('trainProgress').style.display = '';
        document.getElementById('trainResults').style.display = 'none';
        document.getElementById('stopTrainingBtn').style.display = '';
        document.getElementById('startTrainingBtn').disabled = true;
        
        this._trainHistory = { train_loss: [], val_loss: [], train_acc: [], val_acc: [] };
        this._trainChart = new TrainingChart('trainChart');
        this._trainLog = [];
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const event = JSON.parse(line.slice(6));
                    this.handleTrainEvent(event);
                }
            }
        } catch (error) {
            this.showToast('Training failed: ' + error.message, 'error');
            this.resetTrainModal();
        }
    }
    
    async stopTraining() {
        try {
            await fetch('http://localhost:8000/train/stop', { method: 'POST' });
            this.showToast('Training stopped', 'info');
        } catch (e) {}
    }
    
    handleTrainEvent(event) {
        if (event.type === 'device_info') {
            this.addTrainLog(`Device: ${event.device}`);
            if (event.requested === 'cuda' && event.actual === 'cpu') {
                this.showToast('CUDA unavailable, training on CPU. Install PyTorch with CUDA support to use GPU.', 'warning');
            }
        }
        
        if (event.type === 'progress') {
            document.getElementById('trainEpochLabel').textContent = `Epoch ${event.epoch}/${event.total_epochs}`;
            document.getElementById('trainTimeLabel').textContent = this.formatTime(event.elapsed);
            document.getElementById('trainProgressBar').style.width = event.progress + '%';
            document.getElementById('metricTrainLoss').textContent = event.train_loss.toFixed(4);
            document.getElementById('metricTrainAcc').textContent = event.train_acc.toFixed(1) + '%';
        }
        
        if (event.type === 'epoch_end') {
            document.getElementById('trainEpochLabel').textContent = `Epoch ${event.epoch}/${event.total_epochs}`;
            document.getElementById('trainTimeLabel').textContent = this.formatTime(event.elapsed);
            document.getElementById('trainProgressBar').style.width = ((event.epoch / event.total_epochs) * 100) + '%';
            document.getElementById('metricTrainLoss').textContent = event.train_loss.toFixed(4);
            document.getElementById('metricValLoss').textContent = event.val_loss.toFixed(4);
            document.getElementById('metricTrainAcc').textContent = event.train_acc.toFixed(1) + '%';
            document.getElementById('metricValAcc').textContent = event.val_acc.toFixed(1) + '%';
            
            this._trainHistory = event.history;
            this._trainChart.update(this._trainHistory);
            
            this.addTrainLog(`Epoch ${event.epoch}/${event.total_epochs} | Loss: ${event.train_loss.toFixed(4)} | Val Loss: ${event.val_loss.toFixed(4)} | Acc: ${event.train_acc.toFixed(1)}% | Val Acc: ${event.val_acc.toFixed(1)}%`);
        }
        
        if (event.type === 'complete') {
            document.getElementById('stopTrainingBtn').style.display = 'none';
            document.getElementById('trainProgress').style.display = 'none';
            document.getElementById('trainResults').style.display = '';
            document.getElementById('resultTrainLoss').textContent = event.final_train_loss.toFixed(4);
            document.getElementById('resultValLoss').textContent = event.final_val_loss.toFixed(4);
            document.getElementById('resultTrainAcc').textContent = event.final_train_acc.toFixed(1) + '%';
            document.getElementById('resultValAcc').textContent = event.final_val_acc.toFixed(1) + '%';
            document.getElementById('resultParams').textContent = event.total_params.toLocaleString();
            document.getElementById('resultTime').textContent = this.formatTime(event.total_time);
            this._weightsFilename = event.weights_path || null;
            this.showToast('Training complete!', 'success');
        }
        
        if (event.type === 'error') {
            document.getElementById('stopTrainingBtn').style.display = 'none';
            this.showToast(event.message, 'error');
            this.addTrainLog(`ERROR: ${event.message}`);
        }
    }
    
    addTrainLog(message) {
        const log = document.getElementById('trainLog');
        const entry = document.createElement('div');
        entry.className = 'train-log-entry';
        entry.textContent = message;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
    }
    
    formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
    
    async exportWeights() {
        if (!this._weightsFilename) {
            this.showToast('No trained model weights available. Train a model first.', 'warning');
            return;
        }
        
        const modelName = document.getElementById('modelName').value || 'NeuralNetwork';
        const url = `http://localhost:8000/weights/${this._weightsFilename}`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error('Weights file not found on server');
            }
            const blob = await response.blob();
            const downloadUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `${modelName.toLowerCase()}_weights.pth`;
            a.click();
            URL.revokeObjectURL(downloadUrl);
            this.showToast('Weights exported successfully!', 'success');
        } catch (error) {
            this.showToast('Failed to export weights: ' + error.message, 'error');
        }
    }
    
    async openWeightsModal() {
        document.getElementById('weightsModal').classList.add('active');
        await this.loadWeightsList();
    }
    
    closeWeightsModal() {
        document.getElementById('weightsModal').classList.remove('active');
    }
    
    async loadWeightsList() {
        const container = document.getElementById('weightsList');
        const emptyState = document.getElementById('weightsEmpty');
        
        try {
            const res = await fetch('http://localhost:8000/weights');
            const data = await res.json();
            const weights = data.weights || [];
            
            if (weights.length === 0) {
                emptyState.style.display = '';
                container.querySelectorAll('.weight-item').forEach(el => el.remove());
                return;
            }
            
            emptyState.style.display = 'none';
            container.querySelectorAll('.weight-item').forEach(el => el.remove());
            
            weights.forEach(w => {
                const item = document.createElement('div');
                item.className = 'weight-item';
                const name = w.filename.replace(/\.pth$/, '').replace(/_\d{8}_\d{6}$/, '');
                const time = new Date(w.modified * 1000).toLocaleString();
                item.innerHTML = `
                    <div class="weight-info">
                        <span class="weight-name">${name}</span>
                        <span class="weight-meta">${w.size_human} · ${time}</span>
                    </div>
                    <div class="weight-actions">
                        <button class="btn btn-sm btn-primary weight-download-btn" data-filename="${w.filename}" title="Download">
                            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                <path d="M7 1v7M4 5l3 3 3-3M2 10v2a1 1 0 001 1h8a1 1 0 001-1v-2" fill="none" stroke="currentColor" stroke-width="1.5"/>
                            </svg>
                            Download
                        </button>
                        <button class="btn btn-sm btn-danger weight-delete-btn" data-filename="${w.filename}" title="Delete">
                            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                <path d="M5.5 5.5A.5.5 0 016 6v6a.5.5 0 01-1 0V6a.5.5 0 01.5-.5zm2.5 0a.5.5 0 01.5.5v6a.5.5 0 01-1 0V6a.5.5 0 01.5-.5zm3 .5a.5.5 0 00-1 0v6a.5.5 0 001 0V6zM14.5 3a1 1 0 00-1-1H13V1a.5.5 0 00-1 0v1H8V1a.5.5 0 00-1 0v1H5.5l-.707-.707A.5.5 0 004 1.5v.5H2.5a.5.5 0 000 1H3v10a1 1 0 001 1h8a1 1 0 001-1V3h.5a.5.5 0 00.5-.5zM5 2h6v1H5V2zM4 3h8v10H4V3z"/>
                            </svg>
                        </button>
                    </div>
                `;
                container.appendChild(item);
            });
            
            container.querySelectorAll('.weight-download-btn').forEach(btn => {
                btn.addEventListener('click', () => this.downloadWeight(btn.dataset.filename));
            });
            
            container.querySelectorAll('.weight-delete-btn').forEach(btn => {
                btn.addEventListener('click', () => this.deleteWeight(btn.dataset.filename));
            });
        } catch (error) {
            this.showToast('Failed to load weights: ' + error.message, 'error');
        }
    }
    
    async downloadWeight(filename) {
        try {
            const response = await fetch(`http://localhost:8000/weights/${filename}`);
            if (!response.ok) {
                throw new Error('Weights file not found');
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
            this.showToast('Weights downloaded!', 'success');
        } catch (error) {
            this.showToast('Failed to download: ' + error.message, 'error');
        }
    }
    
    async deleteWeight(filename) {
        if (!confirm(`Delete ${filename}?`)) return;
        try {
            const response = await fetch(`http://localhost:8000/weights/${filename}`, { method: 'DELETE' });
            const data = await response.json();
            if (data.status === 'deleted') {
                this.showToast('Weights deleted', 'success');
                await this.loadWeightsList();
            } else {
                this.showToast('Failed to delete weights', 'error');
            }
        } catch (error) {
            this.showToast('Failed to delete: ' + error.message, 'error');
        }
    }
    
    async purgeWeights() {
        if (!confirm('Delete all trained weights? This cannot be undone.')) return;
        try {
            const response = await fetch('http://localhost:8000/weights/purge', { method: 'POST' });
            const data = await response.json();
            if (data.status === 'purged') {
                this.showToast('All weights purged', 'success');
                await this.loadWeightsList();
            } else {
                this.showToast('Failed to purge weights', 'error');
            }
        } catch (error) {
            this.showToast('Failed to purge: ' + error.message, 'error');
        }
    }
    
    openEvalModal() {
        if (this.nodeManager.getNodesArray().length === 0) {
            this.showToast('Add layers to the canvas first!', 'warning');
            return;
        }
        document.getElementById('evalConfig').style.display = '';
        document.getElementById('evalResults').style.display = 'none';
        document.getElementById('evalModal').classList.add('active');
    }
    
    closeEvalModal() {
        document.getElementById('evalModal').classList.remove('active');
    }

    closeDatasetModal() {
        document.getElementById('datasetModal').classList.remove('active');
    }
    
    async startEvaluation() {
        const blueprint = this.getBlueprint();
        const config = {
            blueprint,
            input_size: [
                parseInt(document.getElementById('evalInputC').value) || 3,
                parseInt(document.getElementById('evalInputH').value) || 32,
                parseInt(document.getElementById('evalInputW').value) || 32
            ],
            num_classes: parseInt(document.getElementById('evalNumClasses').value) || 10,
            num_samples: parseInt(document.getElementById('evalSamples').value) || 500,
            val_ratio: 0.2,
            loss_function: document.getElementById('evalLoss').value
        };
        
        document.getElementById('startEvalBtn').disabled = true;
        document.getElementById('startEvalBtn').textContent = 'Evaluating...';
        
        try {
            const response = await fetch('http://localhost:8000/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (result.status === 'error') {
                this.showToast(result.message, 'error');
                return;
            }
            
            document.getElementById('evalConfig').style.display = 'none';
            document.getElementById('evalResults').style.display = '';
            document.getElementById('evalAccuracy').textContent = result.val_accuracy.toFixed(1) + '%';
            document.getElementById('evalLoss').textContent = result.val_loss.toFixed(4);
            document.getElementById('evalTotalParams').textContent = result.total_params.toLocaleString();
            document.getElementById('evalTrainableParams').textContent = result.trainable_params.toLocaleString();
            document.getElementById('evalNumClassesResult').textContent = result.num_classes;
            
            const barsContainer = document.getElementById('classBars');
            barsContainer.innerHTML = '';
            
            if (result.per_class_accuracy) {
                result.per_class_accuracy.forEach(stat => {
                    const row = document.createElement('div');
                    row.className = 'class-bar-row';
                    row.innerHTML = `
                        <span class="class-bar-label">Class ${stat.class}</span>
                        <div class="class-bar-track">
                            <div class="class-bar-fill" style="width: ${stat.accuracy}%"></div>
                        </div>
                        <span class="class-bar-value">${stat.accuracy.toFixed(1)}%</span>
                    `;
                    barsContainer.appendChild(row);
                });
            }
            
            this.showToast('Evaluation complete!', 'success');
        } catch (error) {
            this.showToast('Evaluation failed: ' + error.message, 'error');
        } finally {
            document.getElementById('startEvalBtn').disabled = false;
            document.getElementById('startEvalBtn').innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M11.251.068a.5.5 0 01.227.58L9.677 6.5H13a.5.5 0 01.364.843l-8 8.5a.5.5 0 01-.842-.49L6.323 9.5H3a.5.5 0 01-.364-.843l8-8.5a.5.5 0 01.615-.09z"/>
                </svg>
                Evaluate
            `;
        }
    }
}

class TrainingChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }
    
    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.width = rect.width;
        this.height = rect.height;
    }
    
    update(history) {
        this.ctx.clearRect(0, 0, this.width, this.height);
        this.drawChart(history.train_loss, history.val_loss, history.train_acc, history.val_acc);
    }
    
    drawChart(trainLoss, valLoss, trainAcc, valAcc) {
        const padding = { top: 20, right: 20, bottom: 30, left: 50 };
        const chartW = this.width - padding.left - padding.right;
        const chartH = this.height - padding.top - padding.bottom;
        const midY = padding.top + chartH / 2;
        
        this.ctx.strokeStyle = 'rgba(42, 42, 74, 0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([4, 4]);
        this.ctx.beginPath();
        this.ctx.moveTo(padding.left, midY);
        this.ctx.lineTo(padding.left + chartW, midY);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        this.drawLines(trainLoss, valLoss, padding, chartW, chartH, 'loss');
        this.drawLines(trainAcc, valAcc, padding, chartW, chartH, 'acc');
        
        this.ctx.font = '11px Inter, sans-serif';
        this.ctx.fillStyle = '#6366f1';
        this.ctx.fillText('Train Loss', padding.left, 14);
        this.ctx.fillStyle = '#ef4444';
        this.ctx.fillText('Val Loss', padding.left + 80, 14);
        this.ctx.fillStyle = '#22c55e';
        this.ctx.fillText('Train Acc', padding.left + 160, 14);
        this.ctx.fillStyle = '#f59e0b';
        this.ctx.fillText('Val Acc', padding.left + 240, 14);
    }
    
    drawLines(trainData, valData, padding, chartW, chartH, type) {
        if (!trainData || trainData.length < 2) return;
        
        const epochs = trainData.length;
        const allValues = [...trainData, ...valData];
        let minVal = Math.min(...allValues);
        let maxVal = Math.max(...allValues);
        
        if (maxVal === minVal) { maxVal += 1; minVal -= 1; }
        const range = maxVal - minVal;
        
        const getX = (i) => padding.left + (i / (epochs - 1)) * chartW;
        const getY = (v) => padding.top + chartH - ((v - minVal) / range) * chartH;
        
        const colors = { loss: { train: '#6366f1', val: '#ef4444' }, acc: { train: '#22c55e', val: '#f59e0b' } };
        
        this.drawLine(trainData, getX, getY, colors[type].train);
        this.drawLine(valData, getX, getY, colors[type].val);
    }
    
    drawLine(data, getX, getY, color) {
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        data.forEach((v, i) => {
            const x = getX(i);
            const y = getY(v);
            i === 0 ? this.ctx.moveTo(x, y) : this.ctx.lineTo(x, y);
        });
        this.ctx.stroke();
        
        this.ctx.fillStyle = color;
        data.forEach((v, i) => {
            this.ctx.beginPath();
            this.ctx.arc(getX(i), getY(v), 3, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new App();
});
