class ConnectionManager {
    constructor(canvas, nodeManager) {
        this.canvas = canvas;
        this.nodeManager = nodeManager;
        this.connections = [];
        this.selectedConnection = null;
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
            
            const path = this.createPath(startPos.x, startPos.y, endPos.x, endPos.y);
            const isHit = this.isPointOnPath(worldPos.x, worldPos.y, path);
            
            if (isHit) {
                this.selectConnection(index);
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
        this.updateCounts();
        
        if (window.app && window.app.onConnectionsChanged) {
            window.app.onConnectionsChanged();
        }
        
        this.canvas.render();
    }
    
    removeConnectionsForNode(nodeId) {
        this.connections = this.connections.filter(c => c.from_id !== nodeId && c.to_id !== nodeId);
        this.updateCounts();
        this.canvas.render();
    }
    
    updateCounts() {
        document.getElementById('connCount').textContent = `Connections: ${this.connections.length}`;
    }
    
   selectConnection(index) {
        if (this.selectedConnection === index) return;
        
        this.deselectAll();
        this.selectedConnection = index;
        this.canvas.render();
    }
    
    deselectAll() {
        this.selectedConnection = null;
    }
    
  isPointOnPath(worldX, worldY, path) {
        const ctx = this.canvas.ctx;
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.strokeStyle = 'transparent';
        ctx.lineWidth = 12;
        
        const parts = path.trim().split(/\s+/);
        let i = 0;
        while (i < parts.length) {
            if (parts[i] === 'M' && i + 2 < parts.length) {
                ctx.moveTo(parseFloat(parts[i + 1]), parseFloat(parts[i + 2]));
                i += 3;
            } else if (parts[i] === 'C' && i + 6 < parts.length) {
                ctx.bezierCurveTo(
                    parseFloat(parts[i + 1]),
                    parseFloat(parts[i + 2]),
                    parseFloat(parts[i + 3]),
                    parseFloat(parts[i + 4]),
                    parseFloat(parts[i + 5]),
                    parseFloat(parts[i + 6])
                );
                i += 7;
            } else {
                i++;
            }
        }
        ctx.stroke();
        
        const hit = ctx.isPointInPath(worldX, worldY);
        ctx.restore();
        
        return hit;
    }
    
    createPath(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const cp = Math.max(50, dx * 0.5);
        return `M ${x1} ${y1} C ${x1 + cp} ${y1}, ${x2 - cp} ${y2}, ${x2} ${y2}`;
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
        this.updateCounts();
        this.canvas.render();
    }
    
    importConnections(conns) {
        this.connections = conns.map(c => ({ from_id: c.from_id, to_id: c.to_id }));
        this.updateCounts();
        this.canvas.render();
    }
}
