class TrainingManager {
    constructor(app) {
        this.app = app;
        this._trainingActive = false;
        this._trainingState = null; // 'running' | 'complete' | 'error' | null
        this._trainingCompleted = false;
        this._trainHistory = null;
        this._trainChart = null;
        this._trainLog = null;
        this._trainAbortController = null;
        this._weightsFilename = null;
    }

    openTrainModal() {
        if (this.app.nodeManager.getNodesArray().length === 0 && !this._trainingActive && !this._trainingCompleted) {
            this.app.showToast('Add layers to the canvas first!', 'warning');
            return;
        }
        if (this._trainingActive || this._trainingCompleted) {
            this.restoreTrainModal();
        } else {
            this.resetTrainModal();
        }
        this.populateDatasetSelector();
        document.getElementById('trainModal').classList.add('active');
    }

    async populateDatasetSelector() {
        const select = document.getElementById('trainDataset');
        select.value = '';
        try {
            const res = await fetch('http://localhost:8000/datasets');
            const data = await res.json();
            const datasets = data.datasets || [];
            select.innerHTML = '<option value="">Synthetic (random data)</option>';
            datasets.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds.id;
                opt.textContent = `${ds.name} (${ds.num_samples} samples, ${ds.num_classes || '?'} classes)`;
                select.appendChild(opt);
            });
        } catch (e) {}
        this.updateSyntheticFields();
        select.addEventListener('change', () => this.updateSyntheticFields());
    }

    updateSyntheticFields() {
        const datasetId = document.getElementById('trainDataset').value;
        const show = !datasetId;
        document.querySelectorAll('#syntheticFields, #syntheticFields2, #syntheticFields3, #syntheticFields4').forEach(el => {
            el.style.display = show ? '' : 'none';
        });
    }

    closeTrainModal() {
        document.getElementById('trainModal').classList.remove('active');
        if (this._trainingActive && this._trainingState === 'running') {
            this.showTrainingStatusBar();
        }
    }

    setupTrainingStatusBar() {
        const bar = document.getElementById('trainStatusBar');
        bar.addEventListener('click', (e) => {
            if (e.target.closest('.train-status-btn')) return;
            this.restoreTrainModal();
            document.getElementById('trainModal').classList.add('active');
        });
        document.getElementById('trainStatusStopBtn').addEventListener('click', (e) => {
            e.stopPropagation();
            this.stopTraining();
        });
        document.getElementById('trainStatusDismiss').addEventListener('click', (e) => {
            e.stopPropagation();
            this.hideTrainingStatusBar();
        });
    }

    showTrainingStatusBar() {
        const bar = document.getElementById('trainStatusBar');
        bar.className = 'train-status-bar active';
        bar.style.display = '';
    }

    hideTrainingStatusBar() {
        document.getElementById('trainStatusBar').style.display = 'none';
    }

    updateTrainingStatusBar(event) {
        if (event.type === 'progress' || event.type === 'epoch_end') {
            document.getElementById('trainStatusEpoch').textContent =
                `Epoch ${event.epoch}/${event.total_epochs}`;
            document.getElementById('trainStatusTime').textContent = this.formatTime(event.elapsed);
            const pct = event.progress !== undefined
                ? event.progress
                : (event.epoch / event.total_epochs) * 100;
            document.getElementById('trainStatusProgress').style.width = pct + '%';
        }
        if (event.type === 'complete') {
            const bar = document.getElementById('trainStatusBar');
            bar.className = 'train-status-bar complete';
            document.getElementById('trainStatusIcon').innerHTML =
                '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">' +
                '<path d="M7 0a7 7 0 100 14A7 7 0 007 0zm3.29 4.29a.5.5 0 01.71.71l-4.5 4.5a.5.5 0 01-.71 0l-2.5-2.5a.5.5 0 01.71-.71L6 8.29l3.79-3.8a.5.5 0 01.5-.2z"/></svg>';
            document.getElementById('trainStatusText').textContent = 'Training Complete';
            document.getElementById('trainStatusStopBtn').style.display = 'none';
        }
        if (event.type === 'error') {
            const bar = document.getElementById('trainStatusBar');
            bar.className = 'train-status-bar error';
            document.getElementById('trainStatusIcon').innerHTML =
                '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">' +
                '<path d="M7 0a7 7 0 100 14A7 7 0 007 0zm3.54 10.46a.5.5 0 01-.71.71L7 8.21l-2.83 2.83a.5.5 0 01-.71-.71L6.29 7.5 3.46 4.67a.5.5 0 01.71-.71L7 6.79l2.83-2.83a.5.5 0 01.71.71L7.71 7.5l2.83 2.83z"/></svg>';
            document.getElementById('trainStatusText').textContent = 'Training Failed';
            document.getElementById('trainStatusStopBtn').style.display = 'none';
        }
    }

    restoreTrainModal() {
        document.getElementById('trainConfig').style.display =
            (this._trainingActive || this._trainingCompleted) ? 'none' : '';
        document.getElementById('trainProgress').style.display =
            (this._trainingState === 'running') ? '' : 'none';
        document.getElementById('trainResults').style.display =
            (this._trainingState === 'complete') ? '' : 'none';
        document.getElementById('stopTrainingBtn').style.display =
            (this._trainingState === 'running') ? '' : 'none';
    }

    resetTrainModal() {
        document.getElementById('trainConfig').style.display = '';
        document.getElementById('trainProgress').style.display = 'none';
        document.getElementById('trainResults').style.display = 'none';
        document.getElementById('stopTrainingBtn').style.display = 'none';
        document.getElementById('startTrainingBtn').disabled = false;
        this._trainHistory = null;
        this._trainChart = null;
        this._weightsFilename = null;
        this.hideTrainingStatusBar();
    }

    async startTraining() {
        const blueprint = this.app.getBlueprint();
        const datasetId = document.getElementById('trainDataset').value;

        const baseConfig = {
            blueprint,
            epochs: parseInt(document.getElementById('trainEpochs').value) || 10,
            learning_rate: parseFloat(document.getElementById('trainLR').value) || 0.001,
            batch_size: parseInt(document.getElementById('trainBatchSize').value) || 32,
            optimizer: document.getElementById('trainOptimizer').value,
            loss_function: document.getElementById('trainLoss').value,
            scheduler: document.getElementById('trainScheduler').value,
            weight_decay: parseFloat(document.getElementById('trainWeightDecay').value) || 0.0,
            step_size: 30,
            gamma: 0.1,
            val_ratio: 0.2
        };

        let config;
        let url;

        if (datasetId) {
            config = { ...baseConfig, dataset_id: datasetId };
            url = 'http://localhost:8000/train/dataset';
        } else {
            config = {
                ...baseConfig,
                input_size: [
                    parseInt(document.getElementById('trainInputC').value) || 3,
                    parseInt(document.getElementById('trainInputH').value) || 32,
                    parseInt(document.getElementById('trainInputW').value) || 32
                ],
                num_classes: parseInt(document.getElementById('trainNumClasses').value) || 10,
                num_samples: parseInt(document.getElementById('trainSamples').value) || 1000,
            };
            url = 'http://localhost:8000/train';
        }

        document.getElementById('trainConfig').style.display = 'none';
        document.getElementById('trainProgress').style.display = '';
        document.getElementById('trainResults').style.display = 'none';
        document.getElementById('stopTrainingBtn').style.display = '';
        document.getElementById('startTrainingBtn').disabled = true;

        this._trainHistory = { train_loss: [], val_loss: [], train_acc: [], val_acc: [] };
        this._trainChart = new TrainingChart('trainChart');
        this._trainLog = [];

        this._trainingActive = true;
        this._trainingState = 'running';
        this._trainingCompleted = false;
        this._trainAbortController = new AbortController();

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
                signal: this._trainAbortController.signal
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const event = JSON.parse(line.slice(6));
                    this.handleTrainEvent(event);
                }
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                return;
            }
            if (this._trainingState === 'complete' || this._trainingCompleted) {
                return;
            }
            this._trainingActive = false;
            this._trainingState = 'error';
            this.app.showToast('Training failed: ' + error.message, 'error');
            this.resetTrainModal();
            this.hideTrainingStatusBar();
        } finally {
            this._trainAbortController = null;
        }
    }

    async stopTraining() {
        if (this._trainAbortController) {
            this._trainAbortController.abort();
        }
        try {
            await fetch('http://localhost:8000/train/stop', { method: 'POST' });
        } catch (e) {}
        this._trainingActive = false;
        this._trainingState = null;
        this._trainingCompleted = false;
        this.hideTrainingStatusBar();
        this.resetTrainModal();
        this.app.showToast('Training stopped', 'info');
    }

    handleTrainEvent(event) {
        if (event.type === 'device_info') {
            this.addTrainLog(`Device: ${event.device}`);
            if (event.requested !== 'cpu' && event.actual === 'cpu') {
                const deviceType = event.device_type || event.requested;
                const deviceLabels = {
                    cuda: 'NVIDIA CUDA',
                    rocm: 'AMD ROCm',
                    xpu: 'Intel XPU',
                    mps: 'Apple MPS'
                };
                const label = deviceLabels[deviceType] || deviceType.toUpperCase();
                this.app.showToast(`${label} unavailable, training on CPU.`, 'warning');
            }
        }

        if (event.type === 'progress') {
            document.getElementById('trainEpochLabel').textContent = `Epoch ${event.epoch}/${event.total_epochs}`;
            document.getElementById('trainTimeLabel').textContent = this.formatTime(event.elapsed);
            document.getElementById('trainProgressBar').style.width = event.progress + '%';
            document.getElementById('metricTrainLoss').textContent = event.train_loss.toFixed(4);
            document.getElementById('metricTrainAcc').textContent = event.train_acc.toFixed(1) + '%';
            this.updateTrainingStatusBar(event);
        }

        if (event.type === 'epoch_end') {
            document.getElementById('trainEpochLabel').textContent = `Epoch ${event.epoch}/${event.total_epochs}`;
            document.getElementById('trainTimeLabel').textContent = this.formatTime(event.elapsed);
            document.getElementById('trainProgressBar').style.width = ((event.epoch / event.total_epochs) * 100) + '%';
            document.getElementById('metricTrainLoss').textContent = event.train_loss.toFixed(4);
            document.getElementById('metricValLoss').textContent = event.val_loss.toFixed(4);
            document.getElementById('metricTrainAcc').textContent = event.train_acc.toFixed(1) + '%';
            document.getElementById('metricValAcc').textContent = event.val_acc.toFixed(1) + '%';

            this._trainHistory = event.history;
            this._trainChart.update(this._trainHistory);

            this.addTrainLog(`Epoch ${event.epoch}/${event.total_epochs} | Loss: ${event.train_loss.toFixed(4)} | Val Loss: ${event.val_loss.toFixed(4)} | Acc: ${event.train_acc.toFixed(1)}% | Val Acc: ${event.val_acc.toFixed(1)}%`);
            this.updateTrainingStatusBar(event);
        }

        if (event.type === 'complete') {
            this._trainingActive = false;
            this._trainingState = 'complete';
            this._trainingCompleted = true;
            document.getElementById('stopTrainingBtn').style.display = 'none';
            document.getElementById('trainProgress').style.display = 'none';
            document.getElementById('trainResults').style.display = '';
            document.getElementById('resultTrainLoss').textContent = event.final_train_loss.toFixed(4);
            document.getElementById('resultValLoss').textContent = event.final_val_loss.toFixed(4);
            document.getElementById('resultTrainAcc').textContent = event.final_train_acc.toFixed(1) + '%';
            document.getElementById('resultValAcc').textContent = event.final_val_acc.toFixed(1) + '%';
            document.getElementById('resultParams').textContent = event.total_params.toLocaleString();
            document.getElementById('resultTime').textContent = this.formatTime(event.total_time);
            this._weightsFilename = event.weights_path || null;
            this.updateTrainingStatusBar(event);
            this.app.showToast('Training complete!', 'success');
        }

        if (event.type === 'stopped') {
            if (!this._trainingActive) return;
            this._trainingActive = false;
            this._trainingState = 'complete';
            this._trainingCompleted = true;
            document.getElementById('stopTrainingBtn').style.display = 'none';
            document.getElementById('trainProgress').style.display = 'none';
            document.getElementById('trainResults').style.display = '';
            if (event.history && event.history.train_loss && event.history.train_loss.length > 0) {
                const last = event.history.train_loss.length - 1;
                document.getElementById('resultTrainLoss').textContent = event.history.train_loss[last].toFixed(4);
                document.getElementById('resultValLoss').textContent = (event.history.val_loss[last] || 0).toFixed(4);
                document.getElementById('resultTrainAcc').textContent = event.history.train_acc[last].toFixed(1) + '%';
                document.getElementById('resultValAcc').textContent = (event.history.val_acc[last] || 0).toFixed(1) + '%';
                this._trainHistory = event.history;
                if (this._trainChart) this._trainChart.update(this._trainHistory);
            } else {
                document.getElementById('resultTrainLoss').textContent = '-';
                document.getElementById('resultValLoss').textContent = '-';
                document.getElementById('resultTrainAcc').textContent = '-';
                document.getElementById('resultValAcc').textContent = '-';
            }
            document.getElementById('resultParams').textContent = '-';
            document.getElementById('resultTime').textContent = this.formatTime(event.total_time || 0);
            this._weightsFilename = null;
        }

        if (event.type === 'error') {
            this._trainingActive = false;
            this._trainingState = 'error';
            document.getElementById('stopTrainingBtn').style.display = 'none';
            this.app.showToast(event.message, 'error');
            this.addTrainLog(`ERROR: ${event.message}`);
            this.updateTrainingStatusBar(event);
        }
    }

    addTrainLog(message) {
        const log = document.getElementById('trainLog');
        const entry = document.createElement('div');
        entry.className = 'train-log-entry';
        entry.textContent = message;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
    }

    formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
}
