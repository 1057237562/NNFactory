const LAYER_DEFAULTS = {
    conv2d: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, dilation: 1, groups: 1, bias: true },
    conv1d: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 1, padding: 1 },
    convtranspose2d: { in_channels: 64, out_channels: 32, kernel_size: 3, stride: 2, padding: 1, output_padding: 1 },
    maxpool2d: { kernel_size: 2, stride: 2, padding: 0 },
    avgpool2d: { kernel_size: 2, stride: 2 },
    adaptive_avgpool2d: { output_size: 1 },
    linear: { in_features: 512, out_features: 10, bias: true },
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

const LAYER_PARAM_FIELDS = {
    conv2d: [
        { key: 'in_channels', label: 'In Channels', type: 'number', default: 3, min: 1 },
        { key: 'out_channels', label: 'Out Channels', type: 'number', default: 64, min: 1 },
        { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 3, min: 1 },
        { key: 'stride', label: 'Stride', type: 'number', default: 1, min: 1 },
        { key: 'padding', label: 'Padding', type: 'number', default: 1, min: 0 },
        { key: 'dilation', label: 'Dilation', type: 'number', default: 1, min: 1 },
        { key: 'groups', label: 'Groups', type: 'number', default: 1, min: 1 },
        { key: 'bias', label: 'Use Bias', type: 'checkbox', default: true }
    ],
    conv1d: [
        { key: 'in_channels', label: 'In Channels', type: 'number', default: 64, min: 1 },
        { key: 'out_channels', label: 'Out Channels', type: 'number', default: 128, min: 1 },
        { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 3, min: 1 },
        { key: 'stride', label: 'Stride', type: 'number', default: 1, min: 1 },
        { key: 'padding', label: 'Padding', type: 'number', default: 1, min: 0 }
    ],
    convtranspose2d: [
        { key: 'in_channels', label: 'In Channels', type: 'number', default: 64, min: 1 },
        { key: 'out_channels', label: 'Out Channels', type: 'number', default: 32, min: 1 },
        { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 3, min: 1 },
        { key: 'stride', label: 'Stride', type: 'number', default: 2, min: 1 },
        { key: 'padding', label: 'Padding', type: 'number', default: 1, min: 0 },
        { key: 'output_padding', label: 'Output Padding', type: 'number', default: 1, min: 0 }
    ],
    maxpool2d: [
        { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 2, min: 1 },
        { key: 'stride', label: 'Stride', type: 'number', default: 2, min: 1 },
        { key: 'padding', label: 'Padding', type: 'number', default: 0, min: 0 }
    ],
    avgpool2d: [
        { key: 'kernel_size', label: 'Kernel Size', type: 'number', default: 2, min: 1 },
        { key: 'stride', label: 'Stride', type: 'number', default: 2, min: 1 }
    ],
    adaptive_avgpool2d: [{ key: 'output_size', label: 'Output Size', type: 'number', default: 1, min: 1 }],
    linear: [
        { key: 'in_features', label: 'In Features', type: 'number', default: 512, min: 1 },
        { key: 'out_features', label: 'Out Features', type: 'number', default: 10, min: 1 },
        { key: 'bias', label: 'Use Bias', type: 'checkbox', default: true }
    ],
    embedding: [
        { key: 'num_embeddings', label: 'Num Embeddings', type: 'number', default: 1000, min: 1 },
        { key: 'embedding_dim', label: 'Embedding Dim', type: 'number', default: 128, min: 1 }
    ],
    batchnorm2d: [{ key: 'num_features', label: 'Num Features', type: 'number', default: 64, min: 1 }],
    batchnorm1d: [{ key: 'num_features', label: 'Num Features', type: 'number', default: 128, min: 1 }],
    layernorm: [{ key: 'normalized_shape', label: 'Normalized Shape', type: 'number', default: 128, min: 1 }],
    groupnorm: [
        { key: 'num_groups', label: 'Num Groups', type: 'number', default: 32, min: 1 },
        { key: 'num_channels', label: 'Num Channels', type: 'number', default: 64, min: 1 }
    ],
    leakyrelu: [{ key: 'negative_slope', label: 'Negative Slope', type: 'number', default: 0.01, min: 0, max: 1, step: 0.01 }],
    softmax: [{ key: 'dim', label: 'Dimension', type: 'number', default: -1 }],
    multiheadattention: [
        { key: 'embed_dim', label: 'Embed Dim', type: 'number', default: 128, min: 1 },
        { key: 'num_heads', label: 'Num Heads', type: 'number', default: 8, min: 1 },
        { key: 'dropout', label: 'Dropout', type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 }
    ],
    transformerencoderlayer: [
        { key: 'd_model', label: 'D Model', type: 'number', default: 128, min: 1 },
        { key: 'nhead', label: 'Num Heads', type: 'number', default: 8, min: 1 },
        { key: 'dim_feedforward', label: 'Dim Feedforward', type: 'number', default: 512, min: 1 },
        { key: 'dropout', label: 'Dropout', type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 },
        { key: 'activation', label: 'Activation', type: 'select', options: ['relu', 'gelu'], default: 'relu' }
    ],
    lstm: [
        { key: 'input_size', label: 'Input Size', type: 'number', default: 128, min: 1 },
        { key: 'hidden_size', label: 'Hidden Size', type: 'number', default: 256, min: 1 },
        { key: 'num_layers', label: 'Num Layers', type: 'number', default: 1, min: 1 },
        { key: 'dropout', label: 'Dropout', type: 'number', default: 0.0, min: 0, max: 1, step: 0.05 },
        { key: 'bidirectional', label: 'Bidirectional', type: 'checkbox', default: false }
    ],
    gru: [
        { key: 'input_size', label: 'Input Size', type: 'number', default: 128, min: 1 },
        { key: 'hidden_size', label: 'Hidden Size', type: 'number', default: 256, min: 1 },
        { key: 'num_layers', label: 'Num Layers', type: 'number', default: 1, min: 1 },
        { key: 'dropout', label: 'Dropout', type: 'number', default: 0.0, min: 0, max: 1, step: 0.05 },
        { key: 'bidirectional', label: 'Bidirectional', type: 'checkbox', default: false }
    ],
    dropout: [{ key: 'p', label: 'Dropout Probability', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05 }],
    dropout2d: [{ key: 'p', label: 'Dropout Probability', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05 }],
    upsample: [
        { key: 'scale_factor', label: 'Scale Factor', type: 'number', default: 2, min: 1 },
        { key: 'mode', label: 'Mode', type: 'select', options: ['nearest', 'bilinear', 'bicubic', 'trilinear'], default: 'nearest' }
    ],
    pixelshuffle: [{ key: 'upscale_factor', label: 'Upscale Factor', type: 'number', default: 2, min: 1 }]
};

function getLayerDefaults(type) {
    return LAYER_DEFAULTS[type] || {};
}

function getLayerParamFields(type) {
    return LAYER_PARAM_FIELDS[type] || [];
}

window.LayerConfig = {
    LAYER_DEFAULTS,
    LAYER_PARAM_FIELDS,
    getLayerDefaults,
    getLayerParamFields
};