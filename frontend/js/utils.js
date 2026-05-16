const LAYER_CATEGORIES = {
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

const LAYER_DISPLAY_NAMES = {
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

// Get category for a layer type
function getLayerCategory(type) {
    return LAYER_CATEGORIES[type] || 'utility';
}

// Get display name for a layer type
function getLayerDisplayName(type) {
    return LAYER_DISPLAY_NAMES[type] || type;
}

// Layer category colors (for node rendering)
const LAYER_CATEGORY_COLORS = {
    conv: '#3b82f6',
    pool: '#22c55e',
    linear: '#a855f7',
    norm: '#f97316',
    activation: '#ec4899',
    transformer: '#06b6d4',
    rnn: '#8b5cf6',
    regularization: '#64748b',
    utility: '#94a3b8'
};

// Get color for a category
function getLayerCategoryColor(category) {
    return LAYER_CATEGORY_COLORS[category] || '#94a3b8';
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Export to global scope
window.Utils = {
    downloadBlob
};

window.LayerUtils = {
    LAYER_CATEGORIES,
    LAYER_DISPLAY_NAMES,
    LAYER_CATEGORY_COLORS,
    getLayerCategory,
    getLayerDisplayName,
    getLayerCategoryColor
};