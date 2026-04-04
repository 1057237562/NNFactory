class ConnectionManager {
    constructor(canvas, nodeManager) {
        this.canvas = canvas;
        this.nodeManager = nodeManager;
        this.connections = [];
        this.svg = document.getElementById('connectionSvg');
        
        this.isDrawing = false;
        this.startNode = null;
        this.startPort = null;
        this.tempX = 0;
        this.tempY = 0;
        this.selectedConnection = null;
        
        this.init();
    }
    
    init() {
        this.svg.addEventListener('mousedown', (e) => this.onSvgMouseDown(e));
        
        document.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('output-port')) {
                e.preventDefault();
                e.stopPropagation();
                this.startDrawing(e);
            }
        });
        
        document.addEventListener('mousemove', (e) => {
            if (this.isDrawing) {
                this.updateTempConnection(e);
            }
        });
        
        document.addEventListener('mouseup', (e) => {
            if (this.isDrawing) {
                this.finishDrawing(e);
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if ((e.key === 'Delete' || e.key === 'Backspace') && this.selectedConnection !== null) {
                this.removeConnection(this.selectedConnection);
                this.selectedConnection = null;
            }
        });
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        this.startNode = e.target.dataset.node;
        this.startPort = 'output';
        
        const node = this.nodeManager.nodes.get(this.startNode);
        const rect = document.getElementById(this.startNode).getBoundingClientRect();
        const containerRect = this.canvas.container.getBoundingClientRect();
        
        this.startX = rect.right - containerRect.left;
        this.startY = rect.top + rect.height / 2 - containerRect.top;
        
        this.tempX = this.startX;
        this.tempY = this.startY;
        
        this.render();
    }
    
    updateTempConnection(e) {
        const containerRect = this.canvas.container.getBoundingClientRect();
        this.tempX = e.clientX - containerRect.left;
        this.tempY = e.clientY - containerRect.top;
        this.render();
    }
    
    finishDrawing(e) {
        this.isDrawing = false;
        
        const target = document.elementFromPoint(e.clientX, e.clientY);
        if (target && target.classList.contains('input-port')) {
            const endNode = target.dataset.node;
            if (endNode !== this.startNode) {
                this.addConnection(this.startNode, endNode);
            }
        }
        
        this.startNode = null;
        this.startPort = null;
        this.render();
    }
    
    addConnection(fromId, toId) {
        const exists = this.connections.some(c => c.from_id === fromId && c.to_id === toId);
        if (exists) return;
        
        this.connections.push({ from_id: fromId, to_id: toId });
        this.updatePortStyles();
        this.updateCounts();
        
        if (window.app && window.app.onConnectionsChanged) {
            window.app.onConnectionsChanged();
        }
    }
    
    removeConnection(index) {
        this.connections.splice(index, 1);
        this.updatePortStyles();
        this.updateCounts();
        this.render();
        
        if (window.app && window.app.onConnectionsChanged) {
            window.app.onConnectionsChanged();
        }
    }
    
    removeConnectionsForNode(nodeId) {
        this.connections = this.connections.filter(c => c.from_id !== nodeId && c.to_id !== nodeId);
        this.updatePortStyles();
        this.updateCounts();
        this.render();
    }
    
    updatePortStyles() {
        document.querySelectorAll('.port').forEach(port => port.classList.remove('connected'));
        
        this.connections.forEach(conn => {
            const fromEl = document.getElementById(conn.from_id);
            const toEl = document.getElementById(conn.to_id);
            if (fromEl) {
                const outPort = fromEl.querySelector('.output-port');
                if (outPort) outPort.classList.add('connected');
            }
            if (toEl) {
                const inPort = toEl.querySelector('.input-port');
                if (inPort) inPort.classList.add('connected');
            }
        });
    }
    
    updateCounts() {
        document.getElementById('connCount').textContent = `Connections: ${this.connections.length}`;
    }
    
    getPortPosition(nodeId, portType) {
        const node = this.nodeManager.nodes.get(nodeId);
        if (!node) return { x: 0, y: 0 };
        
        if (portType === 'input') {
            return { x: node.x, y: node.y + node.height / 2 };
        } else {
            return { x: node.x + node.width, y: node.y + node.height / 2 };
        }
    }
    
    createPath(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const cp = Math.max(50, dx * 0.5);
        return `M ${x1} ${y1} C ${x1 + cp} ${y1}, ${x2 - cp} ${y2}, ${x2} ${y2}`;
    }
    
    render() {
        this.svg.innerHTML = '';
        
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead');
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '7');
        marker.setAttribute('refX', '10');
        marker.setAttribute('refY', '3.5');
        marker.setAttribute('orient', 'auto');
        
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
        polygon.setAttribute('fill', '#6366f1');
        marker.appendChild(polygon);
        defs.appendChild(marker);
        this.svg.appendChild(defs);
        
        this.connections.forEach((conn, index) => {
            const start = this.getPortPosition(conn.from_id, 'output');
            const end = this.getPortPosition(conn.to_id, 'input');
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', this.createPath(start.x, start.y, end.x, end.y));
            path.setAttribute('class', 'connection-path');
            path.setAttribute('marker-end', 'url(#arrowhead)');
            path.dataset.index = index;
            
            path.addEventListener('click', (e) => {
                e.stopPropagation();
                document.querySelectorAll('.connection-path.selected').forEach(p => p.classList.remove('selected'));
                path.classList.add('selected');
                this.selectedConnection = index;
            });
            
            this.svg.appendChild(path);
        });
        
        if (this.isDrawing) {
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', this.createPath(this.startX, this.startY, this.tempX, this.tempY));
            path.setAttribute('class', 'connection-path animating');
            path.setAttribute('stroke-dasharray', '5,5');
            this.svg.appendChild(path);
        }
    }
    
    onSvgMouseDown(e) {
        if (e.target === this.svg) {
            document.querySelectorAll('.connection-path.selected').forEach(p => p.classList.remove('selected'));
            this.selectedConnection = null;
            this.nodeManager.deselectAll();
        }
    }
    
    getConnectionsArray() {
        return [...this.connections];
    }
    
    clear() {
        this.connections = [];
        this.selectedConnection = null;
        this.updatePortStyles();
        this.updateCounts();
        this.render();
    }
    
    importConnections(conns) {
        this.connections = conns.map(c => ({ from_id: c.from_id, to_id: c.to_id }));
        this.updatePortStyles();
        this.updateCounts();
        this.render();
    }
}
