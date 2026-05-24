class App {
    constructor() {
        this.canvas = new CanvasManager();
        this.nodeManager = new NodeManager(this.canvas);
        this.connectionManager = new ConnectionManager(this.canvas, this.nodeManager);
        this.propertiesPanel = new PropertiesPanel();
        this.codeGenerator = new CodeGenerator();
        this.datasetManager = new DatasetManagerUI();
        this.datasetManager.init();
        this.evaluator = new EvaluatorUI();
        this.evaluator.init();

        this._backendOnline = false;
        this._availableDevices = ['cpu'];
        this._backendCheckInterval = null;
        
        this.training = new TrainingManager(this);
        this.weights = new WeightsManager(this);
        
        window.app = this;
        
        this.init();
        this.loadFromLocalStorage();
        this.checkBackend();
    }
    
    loadFromLocalStorage() {
        const saved = localStorage.getItem('nnfactory_autosave');
        if (saved) {
            try {
                const blueprint = JSON.parse(saved);
                this.nodeManager.importNodes(blueprint.layers || []);
                this.connectionManager.importConnections(blueprint.connections || []);

                this._applyBlueprintFields(blueprint);
                this.renderConnections();
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

    async checkBackend() {
        this.setBackendStatus('checking');
        try {
            const res = await fetch('http://localhost:8000/health', { signal: AbortSignal.timeout(3000) });
            if (!res.ok) throw new Error('Not OK');
            this._backendOnline = true;
            this.setBackendStatus('online');
            this.enableBackendButtons(true);
            await this.queryDevices();
        } catch (e) {
            this._backendOnline = false;
            this.setBackendStatus('offline');
            this.enableBackendButtons(false);
            this.filterDeviceDropdown();
        }
        this.scheduleBackendCheck();
    }

    scheduleBackendCheck() {
        if (this._backendCheckInterval) clearInterval(this._backendCheckInterval);
        this._backendCheckInterval = setInterval(() => this.checkBackend(), 30000);
    }

    setBackendStatus(status) {
        const dot = document.getElementById('backendDot');
        const label = document.getElementById('backendLabel');
        if (!dot || !label) return;
        dot.className = 'backend-dot ' + status;
        const statusTexts = {
            online: 'Backend Online',
            offline: 'Backend Offline',
            checking: 'Checking...'
        };
        label.textContent = statusTexts[status] || status;
    }

    enableBackendButtons(enabled) {
        const btnIds = ['trainBtn', 'evaluateBtn', 'datasetBtn'];
        btnIds.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) {
                btn.disabled = !enabled;
                btn.classList.toggle('disabled', !enabled);
                btn.title = enabled
                    ? btn.dataset.originalTitle || ''
                    : 'Backend server unavailable — start the backend to use this feature';
            }
        });
    }

    async queryDevices() {
        try {
            const res = await fetch('http://localhost:8000/devices', { signal: AbortSignal.timeout(3000) });
            const data = await res.json();
            this._availableDevices = (data.devices || [])
                .filter(d => d.available)
                .map(d => d.id);
            this.filterDeviceDropdown();
        } catch (e) {
            this._availableDevices = ['cpu'];
            this.filterDeviceDropdown();
        }
    }

    filterDeviceDropdown() {
        const select = document.getElementById('deviceSelect');
        if (!select) return;

        const allOptions = {
            cpu:  { label: 'CPU',                     backend: 'cpu' },
            cuda: { label: 'GPU (NVIDIA CUDA)',        backend: 'cuda' },
            rocm: { label: 'GPU (AMD ROCm)',           backend: 'rocm' },
            xpu:  { label: 'GPU (Intel XPU)',          backend: 'xpu' },
            mps:  { label: 'GPU (Apple MPS)',          backend: 'mps' },
        };

        const available = this._backendOnline ? this._availableDevices : ['cpu'];
        const currentValue = select.value;

        select.innerHTML = '';
        Object.entries(allOptions).forEach(([value, opt]) => {
            const isAvail = available.includes(value);
            const option = document.createElement('option');
            option.value = value;
            option.textContent = opt.label;
            option.disabled = !isAvail;
            if (isAvail && !currentValue) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        if (currentValue && available.includes(currentValue)) {
            select.value = currentValue;
        } else if (available.length > 0) {
            select.value = available[0];
        }

        if (!this._backendOnline) {
            select.title = 'Backend offline — only CPU available';
        } else {
            select.title = '';
        }
    }

    init() {
        this.setupEventListeners();
        this.setupCategoryToggles();
        this.training.setupTrainingStatusBar();
        this.initStyleSwitch();
        this.renderConnections();
    }
    
    setupEventListeners() {
        document.getElementById('generateBtn').addEventListener('click', () => this.generateCode());
        document.getElementById('validateBtn').addEventListener('click', () => this.validateBlueprint());
        document.getElementById('trainBtn').addEventListener('click', () => this.training.openTrainModal());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.evaluator.open());
        document.getElementById('fileInput').addEventListener('change', (e) => this.importBlueprint(e));
        document.getElementById('clearBtn').addEventListener('click', () => this.clearCanvas());

        const filesBtn = document.getElementById('filesBtn');
        const filesDropdown = document.getElementById('filesDropdown');
        if (filesBtn && filesDropdown) {
            filesBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                filesDropdown.classList.toggle('open');
                filesBtn.closest('.header-dropdown').classList.toggle('open');
            });
            filesDropdown.querySelectorAll('.dropdown-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    e.stopPropagation();
                    filesDropdown.classList.remove('open');
                    filesBtn.closest('.header-dropdown').classList.remove('open');
                    const action = item.dataset.action;
                    if (action === 'export') this.exportBlueprint();
                    else if (action === 'import') document.getElementById('fileInput').click();
                    else if (action === 'weights') this.weights.openWeightsModal();
                });
            });
            document.addEventListener('click', () => {
                filesDropdown.classList.remove('open');
                filesBtn.closest('.header-dropdown').classList.remove('open');
            });
        }
        
        document.getElementById('closeModal').addEventListener('click', () => this.closeCodeModal());
        document.getElementById('closeTrainModal').addEventListener('click', () => this.training.closeTrainModal());
        document.getElementById('closeEvalModal').addEventListener('click', () => this.evaluator.close());
        document.getElementById('closeWeightsModal').addEventListener('click', () => this.weights.closeWeightsModal());
        
        // Close modals on backdrop click — only for overlays inside .code-modal containers
        document.querySelectorAll('.code-modal > .modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', () => {
                this.closeCodeModal();
                this.training.closeTrainModal();
                if (this.evaluator) this.evaluator.close();
                this.closeDatasetModal();
                this.weights.closeWeightsModal();
            });
        });
        
        document.getElementById('copyCode').addEventListener('click', () => this.copyCode());
        document.getElementById('downloadCode').addEventListener('click', () => this.downloadCode());
        document.getElementById('startTrainingBtn').addEventListener('click', () => this.training.startTraining());
        document.getElementById('stopTrainingBtn').addEventListener('click', () => this.training.stopTraining());
        document.getElementById('trainAgainBtn').addEventListener('click', () => this.training.resetTrainModal());
        document.getElementById('exportWeightsBtn').addEventListener('click', () => this.weights.exportWeights());
        document.getElementById('purgeWeightsBtn').addEventListener('click', () => this.weights.purgeWeights());
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeCodeModal();
                this.training.closeTrainModal();
                if (this.evaluator) this.evaluator.close();
                this.closeDatasetModal();
                this.weights.closeWeightsModal();
            }
            if ((e.key === 'Delete' || e.key === 'Backspace') && !this.isInputFocused()) {
                if (this.connectionManager.selectedConnections.size > 0) {
                    e.preventDefault();
                    this.connectionManager.removeSelectedConnections();
                }
                if (this.nodeManager.selectedNodes && this.nodeManager.selectedNodes.size > 0) {
                    e.preventDefault();
                    this.nodeManager.deleteSelectedNodes();
                    this.propertiesPanel.hide();
                } else if (this.nodeManager.selectedNode) {
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

    initStyleSwitch() {
        const saved = localStorage.getItem('nnfactory_theme');
        if (saved === 'flat') {
            this.enableFlatDesign();
        }
        document.getElementById('styleSwitchBtn').addEventListener('click', () => {
            const isFlat = document.documentElement.classList.toggle('theme-flat');
            localStorage.setItem('nnfactory_theme', isFlat ? 'flat' : 'default');
            this.updateStyleSwitchButton(isFlat);
            this.canvas.render();
        });
    }

    enableFlatDesign() {
        document.documentElement.classList.add('theme-flat');
        this.updateStyleSwitchButton(true);
    }

    updateStyleSwitchButton(isFlat) {
        const btn = document.getElementById('styleSwitchBtn');
        const label = document.getElementById('styleSwitchLabel');
        if (!btn || !label) return;
        if (isFlat) {
            label.textContent = 'Classic';
            btn.title = 'Switch to Classic Dark theme';
            btn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <circle cx="8" cy="8" r="3" fill="currentColor"/>
                    <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" stroke="currentColor" stroke-width="1.5" fill="none"/>
                </svg>
                <span class="style-switch-label" id="styleSwitchLabel">Classic</span>
            `;
        } else {
            label.textContent = 'Flat';
            btn.title = 'Switch to Flat Design';
            btn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <circle cx="8" cy="8" r="3" fill="currentColor"/>
                    <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" stroke="currentColor" stroke-width="1.5" fill="none"/>
                </svg>
                <span class="style-switch-label" id="styleSwitchLabel">Flat</span>
            `;
        }
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
    
    _applyBlueprintFields(blueprint) {
        if (blueprint.model_name) {
            const el = document.getElementById('modelName');
            if (el) el.value = blueprint.model_name;
        }
        if (blueprint.use_jit !== undefined) {
            const el = document.getElementById('useJit');
            if (el) el.checked = blueprint.use_jit;
        }
        if (blueprint.use_compile !== undefined) {
            const el = document.getElementById('useCompile');
            if (el) el.checked = blueprint.use_compile;
        }
        if (blueprint.device !== undefined) {
            const el = document.getElementById('deviceSelect');
            if (el) el.value = blueprint.device;
        }
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
        window.Utils.downloadBlob(blob, `${modelName.toLowerCase()}.py`);
        this.showToast('Code downloaded!', 'success');
    }
    
    exportBlueprint() {
        const blueprint = this.getBlueprint();
        const blob = new Blob([JSON.stringify(blueprint, null, 2)], { type: 'application/octet-stream' });
        window.Utils.downloadBlob(blob, `${blueprint.model_name.toLowerCase()}.nn`);
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
                
                this._applyBlueprintFields(blueprint);
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
    
    openEvalModal() {
        if (this.evaluator) this.evaluator.open();
    }

    closeEvalModal() {
        if (this.evaluator) this.evaluator.close();
    }

    closeDatasetModal() {
        document.getElementById('datasetModal').classList.remove('active');
    }
}
window.addEventListener('modals-loaded', () => {
    window.app = new App();
});
