class WeightsManager {
    constructor(app) {
        this.app = app;
    }

    async exportWeights() {
        if (!this.app.training._weightsFilename) {
            this.app.showToast('No trained model weights available. Train a model first.', 'warning');
            return;
        }

        const modelName = document.getElementById('modelName').value || 'NeuralNetwork';
        const url = `http://localhost:8000/weights/${this.app.training._weightsFilename}`;

        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error('Weights file not found on server');
            }
            const blob = await response.blob();
            window.Utils.downloadBlob(blob, `${modelName.toLowerCase()}_weights.pth`);
            this.app.showToast('Weights exported successfully!', 'success');
        } catch (error) {
            this.app.showToast('Failed to export weights: ' + error.message, 'error');
        }
    }

    async openWeightsModal() {
        document.getElementById('weightsModal').classList.add('active');
        await this.loadWeightsList();
    }

    closeWeightsModal() {
        document.getElementById('weightsModal').classList.remove('active');
    }

    async loadWeightsList() {
        const container = document.getElementById('weightsList');
        const emptyState = document.getElementById('weightsEmpty');

        try {
            const res = await fetch('http://localhost:8000/weights');
            const data = await res.json();
            const weights = data.weights || [];

            if (weights.length === 0) {
                emptyState.style.display = '';
                container.querySelectorAll('.weight-item').forEach(el => el.remove());
                return;
            }

            emptyState.style.display = 'none';
            container.querySelectorAll('.weight-item').forEach(el => el.remove());

            weights.forEach(w => {
                const item = document.createElement('div');
                item.className = 'weight-item';
                const name = w.filename.replace(/\.pth$/, '').replace(/_\d{8}_\d{6}$/, '');
                const time = new Date(w.modified * 1000).toLocaleString();
                item.innerHTML = `
                    <div class="weight-info">
                        <span class="weight-name">${name}</span>
                        <span class="weight-meta">${w.size_human} · ${time}</span>
                    </div>
                    <div class="weight-actions">
                        <button class="btn btn-sm btn-primary weight-download-btn" data-filename="${w.filename}" title="Download">
                            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                <path d="M7 1v7M4 5l3 3 3-3M2 10v2a1 1 0 001 1h8a1 1 0 001-1v-2" fill="none" stroke="currentColor" stroke-width="1.5"/>
                            </svg>
                            Download
                        </button>
                        <button class="btn btn-sm btn-danger weight-delete-btn" data-filename="${w.filename}" title="Delete">
                            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                <path d="M5.5 5.5A.5.5 0 016 6v6a.5.5 0 01-1 0V6a.5.5 0 01.5-.5zm2.5 0a.5.5 0 01.5.5v6a.5.5 0 01-1 0V6a.5.5 0 01.5-.5zm3 .5a.5.5 0 00-1 0v6a.5.5 0 001 0V6zM14.5 3a1 1 0 00-1-1H13V1a.5.5 0 00-1 0v1H8V1a.5.5 0 00-1 0v1H5.5l-.707-.707A.5.5 0 004 1.5v.5H2.5a.5.5 0 000 1H3v10a1 1 0 001 1h8a1 1 0 001-1V3h.5a.5.5 0 00.5-.5zM5 2h6v1H5V2zM4 3h8v10H4V3z"/>
                            </svg>
                        </button>
                    </div>
                `;
                container.appendChild(item);
            });

            container.querySelectorAll('.weight-download-btn').forEach(btn => {
                btn.addEventListener('click', () => this.downloadWeight(btn.dataset.filename));
            });

            container.querySelectorAll('.weight-delete-btn').forEach(btn => {
                btn.addEventListener('click', () => this.deleteWeight(btn.dataset.filename));
            });
        } catch (error) {
            this.app.showToast('Failed to load weights: ' + error.message, 'error');
        }
    }

    async downloadWeight(filename) {
        try {
            const response = await fetch(`http://localhost:8000/weights/${filename}`);
            if (!response.ok) {
                throw new Error('Weights file not found');
            }
            const blob = await response.blob();
            window.Utils.downloadBlob(blob, filename);
            this.app.showToast('Weights downloaded!', 'success');
        } catch (error) {
            this.app.showToast('Failed to download: ' + error.message, 'error');
        }
    }

    async deleteWeight(filename) {
        if (!confirm(`Delete ${filename}?`)) return;
        try {
            const response = await fetch(`http://localhost:8000/weights/${filename}`, { method: 'DELETE' });
            const data = await response.json();
            if (data.status === 'deleted') {
                this.app.showToast('Weights deleted', 'success');
                await this.loadWeightsList();
            } else {
                this.app.showToast('Failed to delete weights', 'error');
            }
        } catch (error) {
            this.app.showToast('Failed to delete: ' + error.message, 'error');
        }
    }

    async purgeWeights() {
        if (!confirm('Delete all trained weights? This cannot be undone.')) return;
        try {
            const response = await fetch('http://localhost:8000/weights/purge', { method: 'POST' });
            const data = await response.json();
            if (data.status === 'purged') {
                this.app.showToast('All weights purged', 'success');
                await this.loadWeightsList();
            } else {
                this.app.showToast('Failed to purge weights', 'error');
            }
        } catch (error) {
            this.app.showToast('Failed to purge: ' + error.message, 'error');
        }
    }
}
