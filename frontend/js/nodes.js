class NodeManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.nodes = new Map();
        this.selectedNode = null;
        this.selectedNodes = new Set();
        this.draggingNode = null;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.nodeIdCounter = 0;
        
        this.boxSelectActive = false;
        this.boxStartX = 0;
        this.boxStartY = 0;
        this.boxEndX = 0;
        this.boxEndY = 0;
        
        this.layerCategories = window.LayerUtils.LAYER_CATEGORIES;
        
        this.layerDefaults = window.LayerConfig.LAYER_DEFAULTS;
        
        this.nodeWidth = 160;
        this.nodeHeight = 70;
        this.headerHeight = 24;
        this.portRadius = 6;
        
        this.clickStartX = 0;
        this.clickStartY = 0;
        this.clickStartTime = 0;
        
        this.init();
    }
    
    init() {
        this.canvas.container.addEventListener('dragover', (e) => e.preventDefault());
        this.canvas.container.addEventListener('drop', (e) => this.onDrop(e));
        this.canvas.container.addEventListener('mousedown', (e) => this.onCanvasMouseDown(e));
        this.canvas.container.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
        this.canvas.container.addEventListener('mouseup', (e) => this.onCanvasMouseUp(e));
        
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
            width: this.nodeWidth,
            height: this.nodeHeight
        };
        
        this.nodes.set(nodeId, node);
        
        if (window.app && window.app.onNodesChanged) {
            window.app.onNodesChanged();
        }
        
        this.canvas.render();
        
        return node;
    }
    
    getDisplayName(type) {
        return window.LayerUtils.getLayerDisplayName(type);
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
    
    getNodeColor(category) {
        return window.LayerUtils.getLayerCategoryColor(category);
    }
    
    renderNode(ctx, node) {
        const x = node.x;
        const y = node.y;
        const w = node.width;
        const h = node.height;
        
        const category = this.layerCategories[node.type] || 'utility';
        const headerColor = this.getNodeColor(category);
        const displayName = this.getDisplayName(node.type);
        const paramSummary = this.getParamSummary(node);
        
        const isSelected = this.selectedNodes.has(node);
        const isDragging = this.draggingNode === node;
        
        ctx.save();
        
        if (isSelected || isDragging) {
            ctx.shadowColor = headerColor;
            ctx.shadowBlur = 15;
            ctx.shadowOffsetY = 4;
        }
        
        ctx.fillStyle = '#1e1e2e';
        ctx.strokeStyle = isSelected ? headerColor : '#3f3f46';
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 8);
        ctx.fill();
        ctx.stroke();

        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.fillStyle = headerColor;
        ctx.beginPath();
        ctx.roundRect(x, y, w, this.headerHeight, [8, 8, 0, 0]);
        ctx.fill();
        
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(displayName, x + w / 2, y + this.headerHeight / 2);
        
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.fillStyle = '#94a3b8';
        if (paramSummary) {
            ctx.fillText(paramSummary, x + w / 2, y + this.headerHeight + h / 2);
        }
        
        const inputPortX = x + this.portRadius;
        const inputPortY = y + h / 2;
        this.renderPort(ctx, inputPortX, inputPortY, 'input', headerColor, isSelected);

        const outputPortX = x + w - this.portRadius;
        this.renderPort(ctx, outputPortX, inputPortY, 'output', headerColor, isSelected);
        
        ctx.restore();
    }
    
    renderPort(ctx, x, y, type, headerColor, isSelected) {
        ctx.save();
        
        // Port glow
        if (isSelected) {
            ctx.shadowColor = headerColor;
            ctx.shadowBlur = 10;
        }
        
        // Port circle
        ctx.fillStyle = '#1e1e2e';
        ctx.strokeStyle = headerColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, this.portRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Inner dot
        ctx.fillStyle = headerColor;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();
    }
    
    getPortPosition(nodeId, portType) {
        const node = this.nodes.get(nodeId);
        if (!node) return { x: 0, y: 0 };
        
        if (portType === 'input') {
            return { x: node.x + this.portRadius, y: node.y + node.height / 2 };
        } else {
            return { x: node.x + node.width - this.portRadius, y: node.y + node.height / 2 };
        }
    }
    
    getNodeAtPosition(worldX, worldY) {
        const nodesArray = Array.from(this.nodes.values()).reverse();
        for (const node of nodesArray) {
            if (worldX >= node.x && worldX <= node.x + node.width &&
                worldY >= node.y && worldY <= node.y + node.height) {
                return node;
            }
        }
        return null;
    }
    
    isNearPort(worldX, worldY, nodeId, portType, threshold = 10) {
        const pos = this.getPortPosition(nodeId, portType);
        const dx = worldX - pos.x;
        const dy = worldY - pos.y;
        return Math.sqrt(dx * dx + dy * dy) <= threshold;
    }
    
    onCanvasMouseDown(e) {
        const rect = this.canvas.container.getBoundingClientRect();
        const worldPos = this.canvas.screenToWorld(
            e.clientX - rect.left,
            e.clientY - rect.top
        );
        
        this.clickStartX = worldPos.x;
        this.clickStartY = worldPos.y;
        this.clickStartTime = Date.now();
        
        const clickedNode = this.getNodeAtPosition(worldPos.x, worldPos.y);
        
        if (clickedNode) {
            const outputPos = this.getPortPosition(clickedNode.id, 'output');
            if (Math.abs(worldPos.x - outputPos.x) <= this.portRadius * 2 &&
                Math.abs(worldPos.y - outputPos.y) <= this.portRadius * 2) {
                this.startConnectionDrag(clickedNode.id, 'output', worldPos.x, worldPos.y);
                return;
            }
            
            const inputPos = this.getPortPosition(clickedNode.id, 'input');
            if (Math.abs(worldPos.x - inputPos.x) <= this.portRadius * 2 &&
                Math.abs(worldPos.y - inputPos.y) <= this.portRadius * 2) {
                this.startConnectionDrag(clickedNode.id, 'input', worldPos.x, worldPos.y);
                return;
            }
            
            this.draggingNode = clickedNode;
            this.dragOffsetX = worldPos.x - clickedNode.x;
            this.dragOffsetY = worldPos.y - clickedNode.y;
            
            this.selectNode(clickedNode);
            this.canvas.render();
        } else {
            // Start box selection on empty space
            this.deselectAll();
            this.boxSelectActive = true;
            this.boxStartX = worldPos.x;
            this.boxStartY = worldPos.y;
            this.boxEndX = worldPos.x;
            this.boxEndY = worldPos.y;
        }
    }
    
    onCanvasMouseMove(e) {
        const rect = this.canvas.container.getBoundingClientRect();
        const worldPos = this.canvas.screenToWorld(
            e.clientX - rect.left,
            e.clientY - rect.top
        );
        
        if (this.draggingNode) {
            const newX = worldPos.x - this.dragOffsetX;
            const newY = worldPos.y - this.dragOffsetY;
            this.draggingNode.x = newX;
            this.draggingNode.y = newY;
            this.canvas.render();
        }
        
        if (this.tempConnection) {
            this.tempConnection.endX = worldPos.x;
            this.tempConnection.endY = worldPos.y;
            this.canvas.render();
        }
        
        if (this.boxSelectActive) {
            this.boxEndX = worldPos.x;
            this.boxEndY = worldPos.y;
            this.canvas.render();
        }
    }
    
    onCanvasMouseUp(e) {
        const rect = this.canvas.container.getBoundingClientRect();
        const worldPos = this.canvas.screenToWorld(
            e.clientX - rect.left,
            e.clientY - rect.top
        );
        
        // Handle box selection completion
        if (this.boxSelectActive) {
            this.boxSelectActive = false;
            
            const minX = Math.min(this.boxStartX, this.boxEndX);
            const minY = Math.min(this.boxStartY, this.boxEndY);
            const maxX = Math.max(this.boxStartX, this.boxEndX);
            const maxY = Math.max(this.boxStartY, this.boxEndY);
            
            const boxWidth = maxX - minX;
            const boxHeight = maxY - minY;
            
            // Only select if the box is big enough (avoid accidental micro-drags)
            if (boxWidth > 5 || boxHeight > 5) {
                this.selectedNodes.clear();
                
                for (const node of this.nodes.values()) {
                    // Check AABB intersection between node and selection box
                    if (node.x < maxX && node.x + node.width > minX &&
                        node.y < maxY && node.y + node.height > minY) {
                        this.selectedNodes.add(node);
                    }
                }
                
                if (this.selectedNodes.size > 0) {
                    this.selectedNode = this.selectedNodes.values().next().value;
                    if (window.app && window.app.propertiesPanel) {
                        window.app.propertiesPanel.show(this.selectedNode);
                    }
                } else {
                    this.selectedNode = null;
                    if (window.app && window.app.propertiesPanel) {
                        window.app.propertiesPanel.hide();
                    }
                }
            }
            
            
            this.canvas.render();
            this.boxStartX = 0;
            this.boxStartY = 0;
            this.boxEndX = 0;
            this.boxEndY = 0;
            return;
        }
        
        const clickDistance = Math.sqrt(
            Math.pow(worldPos.x - this.clickStartX, 2) + 
            Math.pow(worldPos.y - this.clickStartY, 2)
        );
        const clickDuration = Date.now() - this.clickStartTime;
        
        if (this.draggingNode) {
            this.draggingNode.x = this.canvas.snapToGrid(this.draggingNode.x);
            this.draggingNode.y = this.canvas.snapToGrid(this.draggingNode.y);
            this.draggingNode = null;
            
            if (window.app && window.app.onNodesChanged) {
                window.app.onNodesChanged();
            }
            
            this.canvas.render();
            return;
        }
        
    if (clickDistance < 5 && clickDuration < 200) {
            const clickedNode = this.getNodeAtPosition(worldPos.x, worldPos.y);
            
            if (clickedNode) {
                const pos = this.getPortPosition(clickedNode.id, 'output');
                const distToOutput = Math.sqrt(
                    Math.pow(worldPos.x - pos.x, 2) + Math.pow(worldPos.y - pos.y, 2)
                );
                
                const pos2 = this.getPortPosition(clickedNode.id, 'input');
                const distToInput = Math.sqrt(
                    Math.pow(worldPos.x - pos2.x, 2) + Math.pow(worldPos.y - pos2.y, 2)
                );
                
                if (distToOutput > 15 && distToInput > 15) {
                    this.selectNode(clickedNode);
                    this.canvas.render();
                }
            } else {
                // Clicked on empty space - deselect
                this.deselectAll();
                this.canvas.render();
            }
            return;
        }
        
        if (this.tempConnection) {
            const targetNode = this.getNodeAtPosition(worldPos.x, worldPos.y);
            if (targetNode && targetNode.id !== this.tempConnection.startNode) {
                const startPos = this.getPortPosition(this.tempConnection.startNode, this.tempConnection.startPort);
                const endPos = this.getPortPosition(targetNode.id, 'input');
                const dx = worldPos.x - endPos.x;
                const dy = worldPos.y - endPos.y;
                
                if (Math.sqrt(dx * dx + dy * dy) <= this.portRadius * 3) {
                    if (window.app && window.app.connectionManager) {
                        window.app.connectionManager.addConnection(this.tempConnection.startNode, targetNode.id);
                    }
                }
            }
            
            this.tempConnection = null;
            this.canvas.render();
        }
    }
    
    startConnectionDrag(nodeId, portType, startX, startY) {
        this.tempConnection = {
            startNode: nodeId,
            startPort: portType,
            startX: startX,
            startY: startY,
            endX: startX,
            endY: startY
        };
        this.canvas.render();
    }
    
    selectNode(node) {
        if (this.selectedNode) {
            this.canvas.render();
        }
        
        this.selectedNode = node;
        this.selectedNodes.clear();
        this.selectedNodes.add(node);
        
        if (window.app && window.app.propertiesPanel) {
            window.app.propertiesPanel.show(node);
        }
        
        this.canvas.render();
    }
    
    deselectAll() {
        if (this.selectedNode || this.selectedNodes.size > 0) {
            this.selectedNode = null;
            this.selectedNodes.clear();
            if (window.app && window.app.propertiesPanel) {
                window.app.propertiesPanel.hide();
            }
            this.canvas.render();
        }
    }
    
    deleteNode(nodeId) {
        const node = this.nodes.get(nodeId);
        if (node) {
            this.nodes.delete(nodeId);
            
            if (this.selectedNode && this.selectedNode.id === nodeId) {
                this.selectedNode = null;
            }
            this.selectedNodes.delete(node);
            
            if (window.app && window.app.connectionManager) {
                window.app.connectionManager.removeConnectionsForNode(nodeId);
            }
            
            this.updateCounts();
            
            if (window.app && window.app.onNodesChanged) {
                window.app.onNodesChanged();
            }
            
            this.canvas.render();
        }
    }
    
    deleteSelectedNodes() {
        const nodesToDelete = Array.from(this.selectedNodes);
        for (const node of nodesToDelete) {
            this.deleteNode(node.id);
        }
        this.selectedNodes.clear();
        this.selectedNode = null;
        if (window.app && window.app.propertiesPanel) {
            window.app.propertiesPanel.hide();
        }
    }
    
    updateNodeParams(nodeId, params) {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.params = { ...node.params, ...params };
            
            if (window.app && window.app.onNodesChanged) {
                window.app.onNodesChanged();
            }
            
            this.canvas.render();
        }
    }
    
    updateCounts() {
        document.getElementById('nodeCount').textContent = `Nodes: ${this.nodes.size}`;
    }
    
    getNodesArray() {
        return Array.from(this.nodes.values());
    }
    
    clear() {
        this.nodes.clear();
        this.selectedNode = null;
        this.selectedNodes.clear();
        this.updateCounts();
        
        if (window.app && window.app.onNodesChanged) {
            window.app.onNodesChanged();
        }
        
        this.canvas.render();
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
