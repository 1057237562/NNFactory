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
        const fields = {
            conv2d: [
                { key: 'in_channels', label: 'In Channels', type: 'number', default: 3, min: 1 },
                { key: 'out_channels', label: 'Out Channels', type: 'number', default: 64, min: 1 },
                { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 3, min: 1 },
                { key: 'stride', label: 'Stride', type: 'number', default: 1, min: 1 },
                { key: 'padding', label: 'Padding', type: 'number', default: 1, min: 0 },
                { key: 'dilation', label: 'Dilation', type: 'number', default: 1, min: 1 },
                { key: 'groups', label: 'Groups', type: 'number', default: 1, min: 1 },
                { key: 'bias', label: 'Use Bias', type: 'checkbox', default: true }
            ],
            conv1d: [
                { key: 'in_channels', label: 'In Channels', type: 'number', default: 64, min: 1 },
                { key: 'out_channels', label: 'Out Channels', type: 'number', default: 128, min: 1 },
                { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 3, min: 1 },
                { key: 'stride', label: 'Stride', type: 'number', default: 1, min: 1 },
                { key: 'padding', label: 'Padding', type: 'number', default: 1, min: 0 }
            ],
            convtranspose2d: [
                { key: 'in_channels', label: 'In Channels', type: 'number', default: 64, min: 1 },
                { key: 'out_channels', label: 'Out Channels', type: 'number', default: 32, min: 1 },
                { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 3, min: 1 },
                { key: 'stride', label: 'Stride', type: 'number', default: 2, min: 1 },
                { key: 'padding', label: 'Padding', type: 'number', default: 1, min: 0 },
                { key: 'output_padding', label: 'Output Padding', type: 'number', default: 1, min: 0 }
            ],
            maxpool2d: [
                { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 2, min: 1 },
                { key: 'stride', label: 'Stride', type: 'number', default: 2, min: 1 },
                { key: 'padding', label: 'Padding', type: 'number', default: 0, min: 0 }
            ],
            avgpool2d: [
                { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 2, min: 1 },
                { key: 'stride', label: 'Stride', type: 'number', default: 2, min: 1 }
            ],
            adaptive_avgpool2d: [
                { key: 'output_size', label: 'Output Size', type: 'number', default: 1, min: 1 }
            ],
            linear: [
                { key: 'in_features', label: 'In Features', type: 'number', default: 512, min: 1 },
                { key: 'out_features', label: 'Out Features', type: 'number', default: 10, min: 1 },
                { key: 'bias', label: 'Use Bias', type: 'checkbox', default: true }
            ],
            embedding: [
                { key: 'num_embeddings', label: 'Num Embeddings', type: 'number', default: 1000, min: 1 },
                { key: 'embedding_dim', label: 'Embedding Dim', type: 'number', default: 128, min: 1 }
            ],
            batchnorm2d: [
                { key: 'num_features', label: 'Num Features', type: 'number', default: 64, min: 1 }
            ],
            batchnorm1d: [
                { key: 'num_features', label: 'Num Features', type: 'number', default: 128, min: 1 }
            ],
            layernorm: [
                { key: 'normalized_shape', label: 'Normalized Shape', type: 'number', default: 128, min: 1 }
            ],
            groupnorm: [
                { key: 'num_groups', label: 'Num Groups', type: 'number', default: 32, min: 1 },
                { key: 'num_channels', label: 'Num Channels', type: 'number', default: 64, min: 1 }
            ],
            leakyrelu: [
                { key: 'negative_slope', label: 'Negative Slope', type: 'number', default: 0.01, min: 0, max: 1, step: 0.01 }
            ],
            softmax: [
                { key: 'dim', label: 'Dimension', type: 'number', default: -1 }
            ],
            multiheadattention: [
                { key: 'embed_dim', label: 'Embed Dim', type: 'number', default: 128, min: 1 },
                { key: 'num_heads', label: 'Num Heads', type: 'number', default: 8, min: 1 },
                { key: 'dropout', label: 'Dropout', type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 }
            ],
            transformerencoderlayer: [
                { key: 'd_model', label: 'D Model', type: 'number', default: 128, min: 1 },
                { key: 'nhead', label: 'Num Heads', type: 'number', default: 8, min: 1 },
                { key: 'dim_feedforward', label: 'Dim Feedforward', type: 'number', default: 512, min: 1 },
                { key: 'dropout', label: 'Dropout', type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 },
                { key: 'activation', label: 'Activation', type: 'select', options: ['relu', 'gelu'], default: 'relu' }
            ],
            lstm: [
                { key: 'input_size', label: 'Input Size', type: 'number', default: 128, min: 1 },
                { key: 'hidden_size', label: 'Hidden Size', type: 'number', default: 256, min: 1 },
                { key: 'num_layers', label: 'Num Layers', type: 'number', default: 1, min: 1 },
                { key: 'dropout', label: 'Dropout', type: 'number', default: 0.0, min: 0, max: 1, step: 0.05 },
                { key: 'bidirectional', label: 'Bidirectional', type: 'checkbox', default: false }
            ],
            gru: [
                { key: 'input_size', label: 'Input Size', type: 'number', default: 128, min: 1 },
                { key: 'hidden_size', label: 'Hidden Size', type: 'number', default: 256, min: 1 },
                { key: 'num_layers', label: 'Num Layers', type: 'number', default: 1, min: 1 },
                { key: 'dropout', label: 'Dropout', type: 'number', default: 0.0, min: 0, max: 1, step: 0.05 },
                { key: 'bidirectional', label: 'Bidirectional', type: 'checkbox', default: false }
            ],
            dropout: [
                { key: 'p', label: 'Dropout Probability', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05 }
            ],
            dropout2d: [
                { key: 'p', label: 'Dropout Probability', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05 }
            ],
            upsample: [
                { key: 'scale_factor', label: 'Scale Factor', type: 'number', default: 2, min: 1 },
                { key: 'mode', label: 'Mode', type: 'select', options: ['nearest', 'bilinear', 'bicubic', 'trilinear'], default: 'nearest' }
            ],
            pixelshuffle: [
                { key: 'upscale_factor', label: 'Upscale Factor', type: 'number', default: 2, min: 1 }
            ]
        };
        
        return fields[type] || [];
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
        const map = {
            conv2d: 'conv', conv1d: 'conv', convtranspose2d: 'conv',
            maxpool2d: 'pool', avgpool2d: 'pool', adaptive_avgpool2d: 'pool',
            linear: 'linear', embedding: 'linear',
            batchnorm2d: 'norm', batchnorm1d: 'norm', layernorm: 'norm', groupnorm: 'norm',
            relu: 'activation', leakyrelu: 'activation', gelu: 'activation',
            sigmoid: 'activation', tanh: 'activation', softmax: 'activation', silu: 'activation',
            multiheadattention: 'transformer', transformerencoderlayer: 'transformer',
            lstm: 'rnn', gru: 'rnn',
            dropout: 'regularization', dropout2d: 'regularization',
            input: 'utility', flatten: 'utility', upsample: 'utility', pixelshuffle: 'utility', output: 'utility'
        };
        return map[type] || 'utility';
    }
    
    getDisplayName(type) {
        const names = {
            conv2d: 'Conv2d', conv1d: 'Conv1d', convtranspose2d: 'ConvTranspose2d',
            maxpool2d: 'MaxPool2d', avgpool2d: 'AvgPool2d', adaptive_avgpool2d: 'AdaptiveAvgPool2d',
            linear: 'Linear', embedding: 'Embedding',
            batchnorm2d: 'BatchNorm2d', batchnorm1d: 'BatchNorm1d', layernorm: 'LayerNorm', groupnorm: 'GroupNorm',
            relu: 'ReLU', leakyrelu: 'LeakyReLU', gelu: 'GELU',
            sigmoid: 'Sigmoid', tanh: 'Tanh', softmax: 'Softmax', silu: 'SiLU',
            multiheadattention: 'MultiheadAttention', transformerencoderlayer: 'TransformerEncoderLayer',
            lstm: 'LSTM', gru: 'GRU',
            dropout: 'Dropout', dropout2d: 'Dropout2d',
            input: 'Input', flatten: 'Flatten', upsample: 'Upsample', pixelshuffle: 'PixelShuffle', output: 'Output'
        };
        return names[type] || type;
    }
}
