class ConnectionManager {
    constructor(canvas, nodeManager) {
        this.canvas = canvas;
        this.nodeManager = nodeManager;
        this.connections = [];
        this.selectedConnection = null;
        this.selectedConnections = new Set();
        this.draggingConnection = null;
        
        this.init();
    }
    
   init() {
        this.canvas.container.addEventListener('mousedown', (e) => this.onCanvasMouseDown(e));
        this.canvas.container.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
        this.canvas.container.addEventListener('mouseup', (e) => this.onCanvasMouseUp(e));
    }
    
    onCanvasMouseDown(e) {
        const rect = this.canvas.container.getBoundingClientRect();
        const worldPos = this.canvas.screenToWorld(
            e.clientX - rect.left,
            e.clientY - rect.top
        );
        
        for (const [index, conn] of this.connections.entries()) {
            const startPos = this.nodeManager.getPortPosition(conn.from_id, 'output');
            const endPos = this.nodeManager.getPortPosition(conn.to_id, 'input');
            
            const isHit = this.isPointOnPath(worldPos.x, worldPos.y, startPos.x, startPos.y, endPos.x, endPos.y);
            
            if (isHit) {
                if (e.shiftKey) {
                    this.toggleConnection(index);
                } else {
                    this.selectConnection(index);
                }
                return;
            }
        }
        
        this.deselectAll();
        this.canvas.render();
    }
    
    onCanvasMouseMove(e) {
        if (!this.draggingConnection) return;
        
        const rect = this.canvas.container.getBoundingClientRect();
        const worldPos = this.canvas.screenToWorld(
            e.clientX - rect.left,
            e.clientY - rect.top
        );
        
        this.draggingConnection.endX = worldPos.x;
        this.draggingConnection.endY = worldPos.y;
        this.canvas.render();
    }
    
    onCanvasMouseUp(e) {
        if (!this.draggingConnection) return;
        
        const targetNode = this.nodeManager.getNodeAtPosition(this.draggingConnection.endX, this.draggingConnection.endY);
        
        if (targetNode && targetNode.id !== this.draggingConnection.startNode) {
            const endPos = this.nodeManager.getPortPosition(targetNode.id, 'input');
            const dx = this.draggingConnection.endX - endPos.x;
            const dy = this.draggingConnection.endY - endPos.y;
            
            if (Math.sqrt(dx * dx + dy * dy) <= 20) {
                this.addConnection(this.draggingConnection.startNode, targetNode.id);
            }
        }
        
        this.draggingConnection = null;
        this.canvas.render();
    }
    
    addConnection(fromId, toId) {
        const exists = this.connections.some(c => c.from_id === fromId && c.to_id === toId);
        if (exists) return;
        
        this.connections.push({ from_id: fromId, to_id: toId });
        this.updateCounts();
        
        if (window.app && window.app.onConnectionsChanged) {
            window.app.onConnectionsChanged();
        }
        
        this.canvas.render();
    }
    
    removeConnection(index) {
        this.connections.splice(index, 1);
        this.selectedConnections.delete(index);
        this.updateCounts();
        
        if (window.app && window.app.onConnectionsChanged) {
            window.app.onConnectionsChanged();
        }
        
        this.canvas.render();
    }
    
    removeSelectedConnections() {
        const indices = [...this.selectedConnections].sort((a, b) => b - a);
        for (const idx of indices) {
            this.connections.splice(idx, 1);
        }
        this.selectedConnections.clear();
        this.selectedConnection = null;
        this.updateCounts();
        
        if (window.app && window.app.onConnectionsChanged) {
            window.app.onConnectionsChanged();
        }
        
        this.canvas.render();
    }
    
    removeConnectionsForNode(nodeId) {
        this.connections = this.connections.filter(c => c.from_id !== nodeId && c.to_id !== nodeId);
        this.selectedConnection = null;
        this.selectedConnections.clear();
        this.updateCounts();
        this.canvas.render();
    }
    
    updateCounts() {
        document.getElementById('connCount').textContent = `Connections: ${this.connections.length}`;
    }
    
   selectConnection(index) {
        if (this.selectedConnection === index && this.selectedConnections.size === 1 && this.selectedConnections.has(index)) {
            return;
        }
        
        this.deselectAll();
        this.selectedConnection = index;
        this.selectedConnections.add(index);
        this.canvas.render();
    }
    
    toggleConnection(index) {
        if (this.selectedConnections.has(index)) {
            this.selectedConnections.delete(index);
        } else {
            this.selectedConnections.add(index);
        }
        this.selectedConnection = this.selectedConnections.size > 0
            ? this.selectedConnections.values().next().value
            : null;
        this.canvas.render();
    }
    
    deselectAll() {
        this.selectedConnection = null;
        this.selectedConnections.clear();
    }
    
    selectConnectionsInBox(minX, minY, maxX, maxY) {
        for (const [index, conn] of this.connections.entries()) {
            if (this.isConnectionInBox(conn, minX, maxX, minY, maxY)) {
                this.selectedConnections.add(index);
            }
        }
        if (this.selectedConnections.size > 0) {
            this.selectedConnection = this.selectedConnections.values().next().value;
        }
    }
    
    isConnectionInBox(conn, minX, maxX, minY, maxY) {
        const start = this.nodeManager.getPortPosition(conn.from_id, 'output');
        const end = this.nodeManager.getPortPosition(conn.to_id, 'input');
        const cp1x = start.x + 50;
        const cp1y = start.y;
        const cp2x = end.x - 50;
        const cp2y = end.y;
        
        for (let t = 0; t <= 1; t += 0.1) {
            const mt = 1 - t;
            const x = mt * mt * mt * start.x
                + 3 * mt * mt * t * cp1x
                + 3 * mt * t * t * cp2x
                + t * t * t * end.x;
            const y = mt * mt * mt * start.y
                + 3 * mt * mt * t * cp1y
                + 3 * mt * t * t * cp2y
                + t * t * t * end.y;
            if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
                return true;
            }
        }
        return false;
    }
    
   isPointOnPath(worldX, worldY, x1, y1, x2, y2) {
        const cp1x = x1 + 50;
        const cp1y = y1;
        const cp2x = x2 - 50;
        const cp2y = y2;
        
        const threshold = 12 / this.canvas.scale;
        
        let minDist = Infinity;
        const steps = 20;
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const mt = 1 - t;
            const bx = mt * mt * mt * x1
                + 3 * mt * mt * t * cp1x
                + 3 * mt * t * t * cp2x
                + t * t * t * x2;
            const by = mt * mt * mt * y1
                + 3 * mt * mt * t * cp1y
                + 3 * mt * t * t * cp2y
                + t * t * t * y2;
            const dx = worldX - bx;
            const dy = worldY - by;
            const dist = dx * dx + dy * dy;
            if (dist < minDist) minDist = dist;
        }
        
        return Math.sqrt(minDist) <= threshold;
    }
    
    render() {
        this.canvas.render();
    }
    
    getConnectionsArray() {
        return [...this.connections];
    }
    
    clear() {
        this.connections = [];
        this.selectedConnection = null;
        this.selectedConnections.clear();
        this.updateCounts();
        this.canvas.render();
    }
    
    importConnections(conns) {
        this.connections = conns.map(c => ({ from_id: c.from_id, to_id: c.to_id }));
        this.selectedConnection = null;
        this.selectedConnections.clear();
        this.updateCounts();
        this.canvas.render();
    }
}
