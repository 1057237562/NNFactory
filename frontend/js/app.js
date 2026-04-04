class App {
    constructor() {
        this.canvas = new CanvasManager();
        this.nodeManager = new NodeManager(this.canvas);
        this.connectionManager = new ConnectionManager(this.canvas, this.nodeManager);
        this.propertiesPanel = new PropertiesPanel();
        this.codeGenerator = new CodeGenerator();
        
        window.app = this;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupCategoryToggles();
        this.renderConnections();
    }
    
    setupEventListeners() {
        document.getElementById('generateBtn').addEventListener('click', () => this.generateCode());
        document.getElementById('validateBtn').addEventListener('click', () => this.validateBlueprint());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportBlueprint());
        document.getElementById('importBtn').addEventListener('click', () => document.getElementById('fileInput').click());
        document.getElementById('fileInput').addEventListener('change', (e) => this.importBlueprint(e));
        document.getElementById('clearBtn').addEventListener('click', () => this.clearCanvas());
        
        document.getElementById('closeModal').addEventListener('click', () => this.closeCodeModal());
        document.querySelector('.modal-overlay').addEventListener('click', () => this.closeCodeModal());
        document.getElementById('copyCode').addEventListener('click', () => this.copyCode());
        document.getElementById('downloadCode').addEventListener('click', () => this.downloadCode());
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeCodeModal();
            if ((e.key === 'Delete' || e.key === 'Backspace') && !this.isInputFocused()) {
                if (this.nodeManager.selectedNode) {
                    this.nodeManager.deleteNode(this.nodeManager.selectedNode.id);
                    this.propertiesPanel.hide();
                }
            }
        });
        
        this.canvas.container.addEventListener('click', (e) => {
            if (e.target === this.canvas.container || e.target === this.canvas.canvas) {
                this.nodeManager.deselectAll();
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
    }
    
    onConnectionsChanged() {
        this.renderConnections();
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
            device: 'cpu'
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
}

document.addEventListener('DOMContentLoaded', () => {
    new App();
});
