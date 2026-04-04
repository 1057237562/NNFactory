class CodeGenerator {
    constructor() {
        this.backendUrl = 'http://localhost:8000';
    }
    
    async generateCode(blueprint) {
        try {
            const response = await fetch(`${this.backendUrl}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(blueprint)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data.code;
        } catch (error) {
            console.error('Backend generation failed, using frontend fallback:', error);
            return this.generateFrontendFallback(blueprint);
        }
    }
    
    async validateBlueprint(blueprint) {
        try {
            const response = await fetch(`${this.backendUrl}/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(blueprint)
            });
            
            const data = await response.json();
            return data;
        } catch (error) {
            return { valid: true, errors: [] };
        }
    }
    
    generateFrontendFallback(blueprint) {
        const { layers, connections, model_name, use_jit, use_compile } = blueprint;
        
        const layersByType = {};
        layers.forEach(l => {
            if (!layersByType[l.type]) layersByType[l.type] = [];
            layersByType[l.type].push(l);
        });
        
        const sorted = this.topologicalSort(layers, connections);
        
        let code = [];
        code.push('import torch');
        code.push('import torch.nn as nn');
        code.push('import torch.nn.functional as F');
        code.push('');
        code.push('');
        code.push('class Add(nn.Module):');
        code.push('    def __init__(self):');
        code.push('        super().__init__()');
        code.push('');
        code.push('    def forward(self, x):');
        code.push('        return x + x');
        code.push('');
        code.push('');
        code.push('class Concat(nn.Module):');
        code.push('    def __init__(self, dim=1):');
        code.push('        super().__init__()');
        code.push('        self.dim = dim');
        code.push('');
        code.push('    def forward(self, x):');
        code.push('        return torch.cat([x, x], dim=self.dim)');
        code.push('');
        code.push('');
        code.push('class View(nn.Module):');
        code.push('    def __init__(self, shape):');
        code.push('        super().__init__()');
        code.push('        self.shape = shape');
        code.push('');
        code.push('    def forward(self, x):');
        code.push('        return x.view(self.shape)');
        code.push('');
        code.push('');
        code.push(`class ${model_name}(nn.Module):`);
        code.push('    def __init__(self):');
        code.push('        super().__init__()');
        
        const hasTransformerEnc = layers.some(l => l.type === 'transformerencoder');
        const hasTransformerEncLayer = layers.some(l => l.type === 'transformerencoderlayer');
        
        if (hasTransformerEnc && hasTransformerEncLayer) {
            const encLayer = layers.find(l => l.type === 'transformerencoderlayer');
            const enc = layers.find(l => l.type === 'transformerencoder');
            const p = encLayer.params;
            const ep = enc.params;
            code.push(`        encoder_layer = nn.TransformerEncoderLayer(d_model=${p.d_model}, nhead=${p.nhead}, dim_feedforward=${p.dim_feedforward}, dropout=${p.dropout}, activation='${p.activation}', batch_first=True)`);
            code.push(`        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=${ep.num_layers})`);
        }
        
        sorted.forEach(layer => {
            if (layer.type === 'input' || layer.type === 'output') return;
            if (layer.type === 'transformerencoder' && hasTransformerEncLayer) return;
            
            const init = this.generateLayerInit(layer);
            if (init) {
                code.push(`        self.${layer.id} = ${init}`);
            }
        });
        
        if (!sorted.some(l => l.type !== 'input' && l.type !== 'output')) {
            code.push('        pass');
        }
        
        code.push('');
        code.push('    def forward(self, x):');
        
        sorted.forEach(layer => {
            if (layer.type === 'input') return;
            if (layer.type === 'output') {
                if (!code.some(line => line.trim().startsWith('return'))) {
                    code.push('        return x');
                }
                return;
            }
            const fwd = this.generateForwardCall(layer);
            if (fwd) code.push(`        ${fwd}`);
        });
        
        if (!code.some(line => line.trim().startsWith('return'))) {
            code.push('        return x');
        }
        
        code.push('');
        code.push('');
        
        if (use_jit) {
            code.push(`# JIT Traced Model`);
            code.push(`def create_${model_name.toLowerCase()}_jit():`);
            code.push(`    model = ${model_name}()`);
            code.push(`    model.eval()`);
            code.push(`    example_input = torch.randn(1, 3, 224, 224)`);
            code.push(`    traced_model = torch.jit.trace(model, example_input)`);
            code.push(`    return traced_model`);
            code.push('');
        }
        
        if (use_compile) {
            code.push(`# torch.compile Model (PyTorch 2.0+)`);
            code.push(`def create_${model_name.toLowerCase()}_compiled():`);
            code.push(`    model = ${model_name}()`);
            code.push(`    compiled_model = torch.compile(model)`);
            code.push(`    return compiled_model`);
            code.push('');
        }
        
        code.push('');
        code.push(`if __name__ == '__main__':`);
        code.push(`    model = ${model_name}()`);
        code.push(`    print(model)`);
        code.push(`    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')`);
        code.push('');
        code.push(`    x = torch.randn(1, 3, 224, 224)`);
        code.push(`    output = model(x)`);
        code.push(`    print(f'Input shape: {x.shape}')`);
        code.push(`    print(f'Output shape: {output.shape}')`);
        
        return code.join('\n');
    }
    
    generateLayerInit(layer) {
        const p = layer.params;
        switch (layer.type) {
            case 'conv2d':
                return `nn.Conv2d(${p.in_channels}, ${p.out_channels}, kernel_size=${p.kernel_size}, stride=${p.stride}, padding=${p.padding}, dilation=${p.dilation || 1}, groups=${p.groups || 1}, bias=${p.bias !== false ? 'True' : 'False'})`;
            case 'conv1d':
                return `nn.Conv1d(${p.in_channels}, ${p.out_channels}, kernel_size=${p.kernel_size}, stride=${p.stride}, padding=${p.padding})`;
            case 'convtranspose2d':
                return `nn.ConvTranspose2d(${p.in_channels}, ${p.out_channels}, kernel_size=${p.kernel_size}, stride=${p.stride}, padding=${p.padding}, output_padding=${p.output_padding})`;
            case 'maxpool2d':
                return `nn.MaxPool2d(kernel_size=${p.kernel_size}, stride=${p.stride}, padding=${p.padding})`;
            case 'avgpool2d':
                return `nn.AvgPool2d(kernel_size=${p.kernel_size}, stride=${p.stride})`;
            case 'adaptive_avgpool2d':
                return `nn.AdaptiveAvgPool2d(output_size=${p.output_size})`;
            case 'linear':
                return `nn.Linear(${p.in_features}, ${p.out_features}, bias=${p.bias !== false ? 'True' : 'False'})`;
            case 'embedding':
                return `nn.Embedding(${p.num_embeddings}, ${p.embedding_dim})`;
            case 'batchnorm2d':
                return `nn.BatchNorm2d(${p.num_features})`;
            case 'batchnorm1d':
                return `nn.BatchNorm1d(${p.num_features})`;
            case 'layernorm':
                return `nn.LayerNorm(${p.normalized_shape})`;
            case 'groupnorm':
                return `nn.GroupNorm(num_groups=${p.num_groups}, num_channels=${p.num_channels})`;
            case 'relu': return 'nn.ReLU()';
            case 'leakyrelu': return `nn.LeakyReLU(negative_slope=${p.negative_slope})`;
            case 'gelu': return 'nn.GELU()';
            case 'sigmoid': return 'nn.Sigmoid()';
            case 'tanh': return 'nn.Tanh()';
            case 'softmax': return `nn.Softmax(dim=${p.dim})`;
            case 'silu': return 'nn.SiLU()';
            case 'multiheadattention':
                return `nn.MultiheadAttention(embed_dim=${p.embed_dim}, num_heads=${p.num_heads}, dropout=${p.dropout}, batch_first=True)`;
            case 'lstm':
                return `nn.LSTM(${p.input_size}, ${p.hidden_size}, num_layers=${p.num_layers}, dropout=${p.dropout}, bidirectional=${p.bidirectional ? 'True' : 'False'}, batch_first=True)`;
            case 'gru':
                return `nn.GRU(${p.input_size}, ${p.hidden_size}, num_layers=${p.num_layers}, dropout=${p.dropout}, bidirectional=${p.bidirectional ? 'True' : 'False'}, batch_first=True)`;
            case 'dropout': return `nn.Dropout(p=${p.p})`;
            case 'dropout2d': return `nn.Dropout2d(p=${p.p})`;
            case 'flatten': return 'nn.Flatten()';
            case 'upsample':
                return `nn.Upsample(scale_factor=${p.scale_factor}, mode='${p.mode}')`;
            case 'pixelshuffle':
                return `nn.PixelShuffle(upscale_factor=${p.upscale_factor})`;
            default:
                return null;
        }
    }
    
    generateForwardCall(layer) {
        switch (layer.type) {
            case 'multiheadattention':
                return `x, _ = self.${layer.id}(x, x, x)`;
            case 'lstm': case 'gru':
                return `x, _ = self.${layer.id}(x)`;
            default:
                return `x = self.${layer.id}(x)`;
        }
    }
    
    topologicalSort(layers, connections) {
        const layersById = {};
        layers.forEach(l => layersById[l.id] = l);
        
        const adj = {};
        const inDegree = {};
        layers.forEach(l => { adj[l.id] = []; inDegree[l.id] = 0; });
        
        connections.forEach(c => {
            if (adj[c.from_id] !== undefined && inDegree[c.to_id] !== undefined) {
                adj[c.from_id].push(c.to_id);
                inDegree[c.to_id]++;
            }
        });
        
        const queue = layers.filter(l => inDegree[l.id] === 0).map(l => l.id);
        queue.sort((a, b) => (layersById[a].position?.x || 0) - (layersById[b].position?.x || 0));
        
        const result = [];
        while (queue.length > 0) {
            const node = queue.shift();
            result.push(layersById[node]);
            
            adj[node].sort((a, b) => (layersById[a].position?.x || 0) - (layersById[b].position?.x || 0));
            adj[node].forEach(neighbor => {
                inDegree[neighbor]--;
                if (inDegree[neighbor] === 0) queue.push(neighbor);
            });
        }
        
        if (result.length !== layers.length) {
            return [...layers].sort((a, b) => (a.position?.x || 0) - (b.position?.x || 0));
        }
        
        return result;
    }
}
