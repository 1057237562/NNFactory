class PropertiesPanel {
    constructor() {
        this.panel = document.getElementById('propertiesPanel');
        this.content = document.getElementById('propertiesContent');
        this.isVisible = true;
        
        document.getElementById('closeProperties').addEventListener('click', () => this.hide());
    }
    
    show(node) {
        this.panel.classList.remove('hidden');
        this.isVisible = true;
        this.render(node);
    }
    
    hide() {
        this.panel.classList.add('hidden');
        this.isVisible = false;
    }
    
    render(node) {
        this.content.innerHTML = this.generatePropertiesHTML(node);
        this.bindEvents(node);
    }
    
    generatePropertiesHTML(node) {
        const category = this.getCategory(node.type);
        const displayName = this.getDisplayName(node.type);
        
        let html = `
            <div class="property-group">
                <div class="property-group-title">Node Info</div>
                <div class="property-row">
                    <label class="property-label">Type</label>
                    <input type="text" class="property-input" value="${displayName}" disabled>
                </div>
                <div class="property-row">
                    <label class="property-label">ID</label>
                    <input type="text" class="property-input" value="${node.id}" disabled>
                </div>
            </div>
        `;
        
        const params = this.getParamFields(node.type);
        if (params.length > 0) {
            html += `<div class="property-group"><div class="property-group-title">Parameters</div>`;
            
            params.forEach(param => {
                const value = node.params[param.key] !== undefined ? node.params[param.key] : param.default;
                
                if (param.type === 'select') {
                    html += `
                        <div class="property-row">
                            <label class="property-label">${param.label}</label>
                            <select class="property-select" data-param="${param.key}">
                                ${param.options.map(opt => `<option value="${opt}" ${value === opt ? 'selected' : ''}>${opt}</option>`).join('')}
                            </select>
                        </div>
                    `;
                } else if (param.type === 'checkbox') {
                    html += `
                        <div class="property-row">
                            <label class="property-checkbox">
                                <input type="checkbox" data-param="${param.key}" ${value ? 'checked' : ''}>
                                <span>${param.label}</span>
                            </label>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="property-row">
                            <label class="property-label">${param.label}</label>
                            <input type="${param.type || 'number'}" class="property-input" 
                                   data-param="${param.key}" value="${value}" 
                                   ${param.min !== undefined ? `min="${param.min}"` : ''} 
                                   ${param.max !== undefined ? `max="${param.max}"` : ''}
                                   ${param.step !== undefined ? `step="${param.step}"` : ''}>
                        </div>
                    `;
                }
            });
            
            html += `</div>`;
        }
        
        html += `
            <button class="delete-node-btn" id="deleteNodeBtn">
                Delete Node
            </button>
        `;
        
        return html;
    }
    
    getParamFields(type) {
        return window.LayerConfig.getLayerParamFields(type);
    }
    
    bindEvents(node) {
        this.content.querySelectorAll('[data-param]').forEach(el => {
            const eventType = el.type === 'checkbox' ? 'change' : 'input';
            el.addEventListener(eventType, () => {
                const param = el.dataset.param;
                let value;
                if (el.type === 'checkbox') {
                    value = el.checked;
                } else if (el.type === 'number') {
                    value = parseFloat(el.value);
                } else {
                    value = el.value;
                }
                
                if (window.app && window.app.nodeManager) {
                    window.app.nodeManager.updateNodeParams(node.id, { [param]: value });
                }
            });
        });
        
        const deleteBtn = document.getElementById('deleteNodeBtn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                if (window.app && window.app.nodeManager) {
                    window.app.nodeManager.deleteNode(node.id);
                    this.hide();
                }
            });
        }
    }
    
    getCategory(type) {
        return window.LayerUtils.getLayerCategory(type);
    }
    
    getDisplayName(type) {
        return window.LayerUtils.getLayerDisplayName(type);
    }
}
