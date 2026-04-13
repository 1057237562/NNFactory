class CanvasManager {
    constructor() {
        this.canvas = document.getElementById('mainCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.container = document.getElementById('canvasContainer');
        this.wrapper = this.container.querySelector('.canvas-wrapper');
        
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isPanning = false;
        this.lastPanX = 0;
        this.lastPanY = 0;
        
        this.gridSize = 24;
        
        this.init();
    }
    
    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        this.container.addEventListener('mousedown', (e) => this.onMouseDown(e), true);
        this.container.addEventListener('mousemove', (e) => this.onMouseMove(e), true);
        this.container.addEventListener('mouseup', (e) => this.onMouseUp(e), true);
        this.container.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
        this.container.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Prevent canvas mouse events from interfering with properties panel
        const propertiesPanel = document.getElementById('propertiesPanel');
        if (propertiesPanel) {
            propertiesPanel.addEventListener('mousedown', (e) => e.stopPropagation());
            propertiesPanel.addEventListener('mouseup', (e) => e.stopPropagation());
        }
        
      document.getElementById('zoomIn').addEventListener('click', () => this.zoom(0.1));
        document.getElementById('zoomOut').addEventListener('click', () => this.zoom(-0.1));
        document.getElementById('zoomReset').addEventListener('click', () => this.resetZoom());
        
        // Track properties panel size changes to resize canvas
        this.setupPanelResizeObserver();
    }
    
    setupPanelResizeObserver() {
        const propertiesPanel = document.getElementById('propertiesPanel');
        if (!propertiesPanel) return;
        
        const resizeObserver = new ResizeObserver(() => {
            this.resize();
        });
        
        resizeObserver.observe(propertiesPanel);
    }
    
    resize() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.render();
    }
    
    screenToWorld(screenX, screenY) {
        return {
            x: (screenX - this.offsetX) / this.scale,
            y: (screenY - this.offsetY) / this.scale
        };
    }
    
    worldToScreen(worldX, worldY) {
        return {
            x: worldX * this.scale + this.offsetX,
            y: worldY * this.scale + this.offsetY
        };
    }
    
    snapToGrid(value) {
        return Math.round(value / this.gridSize) * this.gridSize;
    }
    
    zoom(delta) {
        const newScale = Math.max(0.25, Math.min(3, this.scale + delta));
        const rect = this.container.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        this.offsetX = centerX - (centerX - this.offsetX) * (newScale / this.scale);
        this.offsetY = centerY - (centerY - this.offsetY) * (newScale / this.scale);
        this.scale = newScale;
        
        this.updateZoomDisplay();
        this.render();
    }
    
    resetZoom() {
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.updateZoomDisplay();
        this.render();
    }
    
    updateZoomDisplay() {
        document.getElementById('zoomLevel').textContent = Math.round(this.scale * 100) + '%';
    }
    
    onMouseDown(e) {
        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            this.isPanning = true;
            this.lastPanX = e.clientX;
            this.lastPanY = e.clientY;
            this.container.style.cursor = 'grabbing';
        }
    }
    
    onMouseMove(e) {
        if (this.isPanning) {
            const dx = e.clientX - this.lastPanX;
            const dy = e.clientY - this.lastPanY;
            this.offsetX += dx;
            this.offsetY += dy;
            this.lastPanX = e.clientX;
            this.lastPanY = e.clientY;
            this.render();
        }
    }
    
    onMouseUp(e) {
        if (this.isPanning) {
            this.isPanning = false;
            this.container.style.cursor = '';
        }
    }
    
    onWheel(e) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -0.05 : 0.05;
        
        const rect = this.container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const newScale = Math.max(0.25, Math.min(3, this.scale + delta));
        const scaleRatio = newScale / this.scale;
        
        this.offsetX = mouseX - (mouseX - this.offsetX) * scaleRatio;
        this.offsetY = mouseY - (mouseY - this.offsetY) * scaleRatio;
        this.scale = newScale;
        
        this.updateZoomDisplay();
        this.render();
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.save();
        this.ctx.translate(this.offsetX, this.offsetY);
        this.ctx.scale(this.scale, this.scale);
        
        this.drawGrid();
        
        if (window.app && window.app.connectionManager) {
            this.drawConnections();
        }
        
        if (window.app && window.app.nodeManager) {
            this.drawNodes();
        }
        
        this.ctx.restore();
    }
    
    drawGrid() {
        const startX = Math.floor(-this.offsetX / this.scale / this.gridSize) * this.gridSize;
        const startY = Math.floor(-this.offsetY / this.scale / this.gridSize) * this.gridSize;
        const endX = startX + (this.canvas.width / this.scale) + this.gridSize * 2;
        const endY = startY + (this.canvas.height / this.scale) + this.gridSize * 2;
        
        this.ctx.fillStyle = 'rgba(42, 42, 74, 0.4)';
        
        for (let x = startX; x < endX; x += this.gridSize) {
            for (let y = startY; y < endY; y += this.gridSize) {
                this.ctx.beginPath();
                this.ctx.arc(x, y, 1, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
    }
    
    drawNodes() {
        const nodes = window.app.nodeManager.getNodesArray();
        for (const node of nodes) {
            window.app.nodeManager.renderNode(this.ctx, node);
        }
    }
    
    drawConnections() {
        const connections = window.app.connectionManager.getConnectionsArray();
        const nodeManager = window.app.nodeManager;
        const ctx = this.ctx;
        
        for (const conn of connections) {
            const start = nodeManager.getPortPosition(conn.from_id, 'output');
            const end = nodeManager.getPortPosition(conn.to_id, 'input');
            
            const path = this.createBezierPath(start.x, start.y, end.x, end.y);
            
            const isSelected = window.app.connectionManager.selectedConnection !== null &&
                connections.indexOf(conn) === window.app.connectionManager.selectedConnection;
            
            ctx.save();
            ctx.strokeStyle = isSelected ? '#6366f1' : '#6366f1';
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.lineCap = 'round';
            ctx.shadowColor = '#6366f1';
            ctx.shadowBlur = isSelected ? 10 : 0;
            
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.bezierCurveTo(start.x + 50, start.y, end.x - 50, end.y, end.x, end.y);
            ctx.stroke();
            
            ctx.shadowBlur = 0;
            
            const arrowSize = 6;
            const angle = Math.atan2(end.y - end.y, end.x - end.x);
            const arrowX = end.x;
            const arrowY = end.y;
            
            ctx.fillStyle = '#6366f1';
            ctx.beginPath();
            ctx.moveTo(arrowX, arrowY);
            ctx.lineTo(arrowX - arrowSize, arrowY - arrowSize / 2);
            ctx.lineTo(arrowX - arrowSize, arrowY + arrowSize / 2);
            ctx.closePath();
            ctx.fill();
            
            ctx.restore();
        }
        
        if (window.app.nodeManager.tempConnection) {
            const temp = window.app.nodeManager.tempConnection;
            const end = nodeManager.getPortPosition(temp.startNode, temp.startPort);
            
            ctx.save();
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            
            ctx.beginPath();
            ctx.moveTo(end.x, end.y);
            ctx.bezierCurveTo(end.x + 50, end.y, temp.endX - 50, temp.endY, temp.endX, temp.endY);
            ctx.stroke();
            
            ctx.restore();
        }
    }
    
    createBezierPath(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const cp = Math.max(50, dx * 0.5);
        return `M ${x1} ${y1} C ${x1 + cp} ${y1}, ${x2 - cp} ${y2}, ${x2} ${y2}`;
    }
    
    getTransform() {
        return { scale: this.scale, offsetX: this.offsetX, offsetY: this.offsetY };
    }
}
