class NodeManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.nodes = new Map();
        this.selectedNode = null;
        this.draggingNode = null;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.nodeIdCounter = 0;
        
        this.layerCategories = {
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
        
        this.layerDefaults = {
            conv2d: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 },
            conv1d: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 1, padding: 1 },
            convtranspose2d: { in_channels: 64, out_channels: 32, kernel_size: 3, stride: 2, padding: 1, output_padding: 1 },
            maxpool2d: { kernel_size: 2, stride: 2, padding: 0 },
            avgpool2d: { kernel_size: 2, stride: 2 },
            adaptive_avgpool2d: { output_size: 1 },
            linear: { in_features: 512, out_features: 10 },
            embedding: { num_embeddings: 1000, embedding_dim: 128 },
            batchnorm2d: { num_features: 64 },
            batchnorm1d: { num_features: 128 },
            layernorm: { normalized_shape: 128 },
            groupnorm: { num_groups: 32, num_channels: 64 },
            relu: {},
            leakyrelu: { negative_slope: 0.01 },
            gelu: {},
            sigmoid: {},
            tanh: {},
            softmax: { dim: -1 },
            silu: {},
            multiheadattention: { embed_dim: 128, num_heads: 8, dropout: 0.1 },
            transformerencoderlayer: { d_model: 128, nhead: 8, dim_feedforward: 512, dropout: 0.1, activation: 'relu' },
            lstm: { input_size: 128, hidden_size: 256, num_layers: 1, dropout: 0.0, bidirectional: false },
            gru: { input_size: 128, hidden_size: 256, num_layers: 1, dropout: 0.0, bidirectional: false },
            dropout: { p: 0.5 },
            dropout2d: { p: 0.5 },
            flatten: {},
            upsample: { scale_factor: 2, mode: 'nearest' },
            pixelshuffle: { upscale_factor: 2 },
            input: {},
            output: {}
        };
        
        this.init();
    }
    
    init() {
        this.canvas.container.addEventListener('dragover', (e) => e.preventDefault());
        this.canvas.container.addEventListener('drop', (e) => this.onDrop(e));
        
        document.querySelectorAll('.layer-item').forEach(item => {
            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('layerType', item.dataset.type);
                item.classList.add('dragging');
            });
            item.addEventListener('dragend', (e) => {
                item.classList.remove('dragging');
            });
        });
        
        document.getElementById('addInputBtn').addEventListener('click', () => {
            const rect = this.canvas.container.getBoundingClientRect();
            this.createNode('input', rect.width / 2 / this.canvas.scale - this.canvas.offsetX / this.canvas.scale, 
                           rect.height / 2 / this.canvas.scale - this.canvas.offsetY / this.canvas.scale);
        });
        
        document.getElementById('addOutputBtn').addEventListener('click', () => {
            const rect = this.canvas.container.getBoundingClientRect();
            this.createNode('output', rect.width / 2 / this.canvas.scale - this.canvas.offsetX / this.canvas.scale + 200, 
                           rect.height / 2 / this.canvas.scale - this.canvas.offsetY / this.canvas.scale);
        });
    }
    
    onDrop(e) {
        e.preventDefault();
        const layerType = e.dataTransfer.getData('layerType');
        if (!layerType) return;
        
        const rect = this.canvas.container.getBoundingClientRect();
        const worldPos = this.canvas.screenToWorld(
            e.clientX - rect.left,
            e.clientY - rect.top
        );
        
        this.createNode(layerType, worldPos.x, worldPos.y);
    }
    
    createNode(type, x, y, params = null, id = null) {
        const nodeId = id || `node_${++this.nodeIdCounter}`;
        const snappedX = this.canvas.snapToGrid(x);
        const snappedY = this.canvas.snapToGrid(y);
        
        const node = {
            id: nodeId,
            type: type,
            params: params || { ...this.layerDefaults[type] },
            x: snappedX,
            y: snappedY,
            width: 160,
            height: 70
        };
        
        this.nodes.set(nodeId, node);
        this.renderNode(node);
        this.updateCounts();
        
        if (window.app && window.app.onNodesChanged) {
            window.app.onNodesChanged();
        }
        
        return node;
    }
    
    renderNode(node) {
        let el = document.getElementById(node.id);
        if (!el) {
            el = document.createElement('div');
            el.id = node.id;
            el.className = 'node';
            el.innerHTML = this.getNodeHTML(node);
            this.canvas.wrapper.appendChild(el);
            
            el.addEventListener('mousedown', (e) => this.onNodeMouseDown(e, node));
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectNode(node);
            });
        }
        
        el.style.left = node.x + 'px';
        el.style.top = node.y + 'px';
        el.style.width = node.width + 'px';
    }
    
    getNodeHTML(node) {
        const category = this.layerCategories[node.type] || 'utility';
        const displayName = this.getDisplayName(node.type);
        const paramSummary = this.getParamSummary(node);
        
        return `
            <div class="node-header ${category}">${displayName}</div>
            <div class="node-body">${paramSummary}</div>
            <div class="node-ports">
                <div class="port input-port" data-node="${node.id}" data-port="input"></div>
                <div class="port output-port" data-node="${node.id}" data-port="output"></div>
            </div>
        `;
    }
    
    getDisplayName(type) {
        const names = {
            conv2d: 'Conv2d', conv1d: 'Conv1d', convtranspose2d: 'ConvTranspose2d',
            maxpool2d: 'MaxPool2d', avgpool2d: 'AvgPool2d', adaptive_avgpool2d: 'AdaptiveAvgPool2d',
            linear: 'Linear', embedding: 'Embedding',
            batchnorm2d: 'BatchNorm2d', batchnorm1d: 'BatchNorm1d', layernorm: 'LayerNorm', groupnorm: 'GroupNorm',
            relu: 'ReLU', leakyrelu: 'LeakyReLU', gelu: 'GELU',
            sigmoid: 'Sigmoid', tanh: 'Tanh', softmax: 'Softmax', silu: 'SiLU',
            multiheadattention: 'MultiheadAttention', transformerencoderlayer: 'TransformerEncLayer',
            lstm: 'LSTM', gru: 'GRU',
            dropout: 'Dropout', dropout2d: 'Dropout2d',
            input: 'Input', flatten: 'Flatten', upsample: 'Upsample', pixelshuffle: 'PixelShuffle', output: 'Output'
        };
        return names[type] || type;
    }
    
    getParamSummary(node) {
        const p = node.params;
        switch (node.type) {
            case 'conv2d': return `${p.in_channels}→${p.out_channels}, k=${p.kernel_size}`;
            case 'conv1d': return `${p.in_channels}→${p.out_channels}, k=${p.kernel_size}`;
            case 'linear': return `${p.in_features}→${p.out_features}`;
            case 'maxpool2d': return `k=${p.kernel_size}, s=${p.stride}`;
            case 'lstm': case 'gru': return `h=${p.hidden_size}, l=${p.num_layers}`;
            case 'dropout': case 'dropout2d': return `p=${p.p}`;
            case 'embedding': return `${p.num_embeddings}×${p.embedding_dim}`;
            case 'multiheadattention': return `d=${p.embed_dim}, h=${p.num_heads}`;
            case 'transformerencoderlayer': return `d=${p.d_model}, h=${p.nhead}`;
            case 'batchnorm2d': case 'batchnorm1d': return `n=${p.num_features}`;
            case 'layernorm': return `n=${p.normalized_shape}`;
            case 'upsample': return `s=${p.scale_factor}`;
            default: return '';
        }
    }
    
    onNodeMouseDown(e, node) {
        if (e.target.classList.contains('port')) return;
        
        e.stopPropagation();
        this.draggingNode = node;
        
        const el = document.getElementById(node.id);
        const rect = el.getBoundingClientRect();
        this.dragOffsetX = e.clientX - rect.left;
        this.dragOffsetY = e.clientY - rect.top;
        
        el.classList.add('dragging');
        el.style.transition = 'none';
        
        const onMouseMove = (e) => {
            if (!this.draggingNode) return;
            
            const containerRect = this.canvas.container.getBoundingClientRect();
            const rawX = (e.clientX - containerRect.left - this.dragOffsetX) / this.canvas.scale;
            const rawY = (e.clientY - containerRect.top - this.dragOffsetY) / this.canvas.scale;
            
            node.x = rawX;
            node.y = rawY;
            
            el.style.left = node.x + 'px';
            el.style.top = node.y + 'px';
            
            if (window.app && window.app.connectionManager) {
                window.app.connectionManager.render();
            }
        };
        
        const onMouseUp = () => {
            el.classList.remove('dragging');
            el.style.transition = '';
            
            node.x = this.canvas.snapToGrid(node.x);
            node.y = this.canvas.snapToGrid(node.y);
            el.style.left = node.x + 'px';
            el.style.top = node.y + 'px';
            
            this.draggingNode = null;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            
            if (window.app && window.app.connectionManager) {
                window.app.connectionManager.render();
            }
        };
        
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    }
    
    selectNode(node) {
        document.querySelectorAll('.node.selected').forEach(el => el.classList.remove('selected'));
        
        this.selectedNode = node;
        const el = document.getElementById(node.id);
        if (el) el.classList.add('selected');
        
        if (window.app && window.app.propertiesPanel) {
            window.app.propertiesPanel.show(node);
        }
    }
    
    deselectAll() {
        document.querySelectorAll('.node.selected').forEach(el => el.classList.remove('selected'));
        this.selectedNode = null;
        if (window.app && window.app.propertiesPanel) {
            window.app.propertiesPanel.hide();
        }
    }
    
    deleteNode(nodeId) {
        const el = document.getElementById(nodeId);
        if (el) el.remove();
        this.nodes.delete(nodeId);
        
        if (this.selectedNode && this.selectedNode.id === nodeId) {
            this.selectedNode = null;
        }
        
        if (window.app && window.app.connectionManager) {
            window.app.connectionManager.removeConnectionsForNode(nodeId);
        }
        
        this.updateCounts();
        
        if (window.app && window.app.onNodesChanged) {
            window.app.onNodesChanged();
        }
    }
    
    updateNodeParams(nodeId, params) {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.params = { ...node.params, ...params };
            const el = document.getElementById(nodeId);
            if (el) {
                el.querySelector('.node-body').textContent = this.getParamSummary(node);
            }
        }
    }
    
    updateCounts() {
        document.getElementById('nodeCount').textContent = `Nodes: ${this.nodes.size}`;
    }
    
    getNodesArray() {
        return Array.from(this.nodes.values());
    }
    
    clear() {
        this.nodes.forEach((node) => {
            const el = document.getElementById(node.id);
            if (el) el.remove();
        });
        this.nodes.clear();
        this.selectedNode = null;
        this.updateCounts();
    }
    
    exportNodes() {
        return Array.from(this.nodes.values()).map(n => ({
            id: n.id,
            type: n.type,
            params: { ...n.params },
            position: { x: n.x, y: n.y }
        }));
    }
    
    importNodes(nodesData) {
        this.clear();
        let maxId = 0;
        nodesData.forEach(n => {
            const numPart = n.id.match(/\d+/);
            if (numPart) maxId = Math.max(maxId, parseInt(numPart[0]));
            this.createNode(n.type, n.position.x, n.position.y, n.params, n.id);
        });
        this.nodeIdCounter = maxId;
    }
}
