class TrainingChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }
    
    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.width = rect.width;
        this.height = rect.height;
    }
    
    update(history) {
        this.ctx.clearRect(0, 0, this.width, this.height);
        this.drawChart(history.train_loss, history.val_loss, history.train_acc, history.val_acc);
    }
    
    drawChart(trainLoss, valLoss, trainAcc, valAcc) {
        const padding = { top: 20, right: 20, bottom: 30, left: 50 };
        const chartW = this.width - padding.left - padding.right;
        const chartH = this.height - padding.top - padding.bottom;
        const midY = padding.top + chartH / 2;
        
        this.ctx.strokeStyle = 'rgba(42, 42, 74, 0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([4, 4]);
        this.ctx.beginPath();
        this.ctx.moveTo(padding.left, midY);
        this.ctx.lineTo(padding.left + chartW, midY);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        this.drawLines(trainLoss, valLoss, padding, chartW, chartH, 'loss');
        this.drawLines(trainAcc, valAcc, padding, chartW, chartH, 'acc');
        
        this.ctx.font = '11px Inter, sans-serif';
        this.ctx.fillStyle = '#6366f1';
        this.ctx.fillText('Train Loss', padding.left, 14);
        this.ctx.fillStyle = '#ef4444';
        this.ctx.fillText('Val Loss', padding.left + 80, 14);
        this.ctx.fillStyle = '#22c55e';
        this.ctx.fillText('Train Acc', padding.left + 160, 14);
        this.ctx.fillStyle = '#f59e0b';
        this.ctx.fillText('Val Acc', padding.left + 240, 14);
    }
    
    drawLines(trainData, valData, padding, chartW, chartH, type) {
        if (!trainData || trainData.length < 2) return;
        
        const epochs = trainData.length;
        const allValues = [...trainData, ...valData];
        let minVal = Math.min(...allValues);
        let maxVal = Math.max(...allValues);
        
        if (maxVal === minVal) { maxVal += 1; minVal -= 1; }
        const range = maxVal - minVal;
        
        const getX = (i) => padding.left + (i / (epochs - 1)) * chartW;
        const getY = (v) => padding.top + chartH - ((v - minVal) / range) * chartH;
        
        const colors = { loss: { train: '#6366f1', val: '#ef4444' }, acc: { train: '#22c55e', val: '#f59e0b' } };
        
        this.drawLine(trainData, getX, getY, colors[type].train);
        this.drawLine(valData, getX, getY, colors[type].val);
    }
    
    drawLine(data, getX, getY, color) {
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        data.forEach((v, i) => {
            const x = getX(i);
            const y = getY(v);
            i === 0 ? this.ctx.moveTo(x, y) : this.ctx.lineTo(x, y);
        });
        this.ctx.stroke();
        
        this.ctx.fillStyle = color;
        data.forEach((v, i) => {
            this.ctx.beginPath();
            this.ctx.arc(getX(i), getY(v), 3, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }
}
