//! Neural Compiler — quantizes weights, generates METL binary, schedules layers
//! Bridges the resolve system output to hardware inference.

use std::collections::HashMap;
use std::io::Write;

/// Quantization mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantMode {
    INT4,
    INT8,
    FP16,
    Mixed,  // per-layer optimal
}

/// A quantized weight tensor
#[derive(Debug, Clone)]
pub struct QuantTensor {
    pub name: String,
    pub original_bits: u32,
    pub quant_bits: u32,
    pub original_size_bytes: usize,
    pub quant_size_bytes: usize,
    pub compression_ratio: f64,
    pub scale: f32,
    pub zero_point: i32,
    pub data: Vec<u8>, // packed quantized data
}

/// A layer in a neural network
#[derive(Debug, Clone)]
pub struct Layer {
    pub id: usize,
    pub name: String,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_size: Option<(usize, usize)>,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub quant_mode: QuantMode,
    pub activation: Activation,
    pub cycles_estimate: u64,
    pub memory_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
    None,
}

impl Layer {
    pub fn weight_count(&self) -> usize {
        self.input_channels * self.output_channels
            * self.kernel_size.map_or(1, |(kh, kw)| kh * kw)
    }
    
    pub fn total_params(&self) -> usize {
        self.weight_count() + self.output_channels // weights + biases
    }
}

/// METL binary format — Metal Binary for inference chips
#[derive(Debug, Clone)]
pub struct MetlHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub num_layers: u16,
    pub input_size: u32,
    pub output_size: u32,
    pub quant_mode: u8,
    pub checksum: u32,
}

impl MetlHeader {
    pub fn new() -> Self {
        Self {
            magic: [b'M', b'E', b'T', b'L'],
            version: 1,
            num_layers: 0,
            input_size: 0,
            output_size: 0,
            quant_mode: QuantMode::INT8 as u8,
            checksum: 0,
        }
    }
    
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.num_layers.to_le_bytes());
        buf.extend_from_slice(&self.input_size.to_le_bytes());
        buf.extend_from_slice(&self.output_size.to_le_bytes());
        buf.push(self.quant_mode);
        buf.extend_from_slice(&[0u8; 3]); // padding
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        buf
    }
}

/// A compiled layer in METL format
#[derive(Debug, Clone)]
pub struct MetlLayer {
    pub layer_id: u16,
    pub op_code: u8,
    pub input_dim: u32,
    pub output_dim: u32,
    pub weight_offset: u32,
    pub weight_size: u32,
    pub bias_offset: u32,
    pub activation: u8,
    pub cycles: u32,
}

impl MetlLayer {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&self.layer_id.to_le_bytes());
        buf.push(self.op_code);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&self.input_dim.to_le_bytes());
        buf.extend_from_slice(&self.output_dim.to_le_bytes());
        buf.extend_from_slice(&self.weight_offset.to_le_bytes());
        buf.extend_from_slice(&self.weight_size.to_le_bytes());
        buf.extend_from_slice(&self.bias_offset.to_le_bytes());
        buf.push(self.activation);
        buf.extend_from_slice(&[0u8; 2]);
        buf.extend_from_slice(&self.cycles.to_le_bytes());
        buf
    }
}

/// Layer schedule — determines execution order on inference chip
#[derive(Debug, Clone)]
pub struct LayerSchedule {
    pub layers: Vec<ScheduledLayer>,
    pub total_cycles: u64,
    pub peak_memory_bytes: usize,
    pub pipeline_depth: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduledLayer {
    pub layer_id: usize,
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub memory_at_start: usize,
    pub can_pipeline: bool,
}

/// The neural compiler
pub struct NeuralCompiler {
    quant_mode: QuantMode,
}

impl NeuralCompiler {
    pub fn new(mode: QuantMode) -> Self { Self { quant_mode: mode } }
    
    /// Quantize a layer's weights
    pub fn quantize_layer(&self, layer: &Layer) -> QuantTensor {
        let bits_per_weight = match self.quant_mode {
            QuantMode::INT4 => 4,
            QuantMode::INT8 => 8,
            QuantMode::FP16 => 16,
            QuantMode::Mixed => {
                // Use INT4 for layers with >1K weights, INT8 otherwise
                if layer.weight_count() > 1024 { 4 } else { 8 }
            }
        };
        
        let bytes_per_weight = (bits_per_weight + 7) / 8;
        let original_bytes = layer.weights.len() * 4; // FP32 original
        
        // Find scale for quantization
        let max_abs = layer.weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let max_val = match bits_per_weight {
            4 => 7.0,   // INT4 range [-8, 7]
            8 => 127.0,  // INT8 range
            16 => max_abs, // FP16
            _ => 127.0,
        };
        let scale = if max_abs > 0.0 { max_abs / max_val } else { 1.0 };
        let zero_point = 0i32;
        
        // Quantize
        let mut packed = Vec::new();
        let mut accum = 0u32;
        let mut accum_bits = 0u32;
        
        for &w in &layer.weights {
            let q = (w / scale).round().clamp(-max_val, max_val) as i32;
            let bits = match bits_per_weight {
                4 => (q as u8) & 0x0F,
                8 => (q as u8),
                16 => {
                    // FP16: store as 2 bytes
                    if accum_bits > 0 { packed.push(accum as u8); accum = 0; accum_bits = 0; }
                    packed.extend_from_slice(&(q as i16).to_le_bytes());
                    continue;
                }
                _ => (q as u8),
            };
            accum |= (bits as u32) << accum_bits;
            accum_bits += bits_per_weight;
            if accum_bits >= 8 {
                packed.push(accum as u8);
                accum >>= 8;
                accum_bits -= 8;
            }
        }
        if accum_bits > 0 { packed.push(accum as u8); }
        
        let quant_bytes = packed.len();
        let compression = original_bytes as f64 / quant_bytes.max(1) as f64;
        
        QuantTensor {
            name: layer.name.clone(),
            original_bits: 32,
            quant_bits: bits_per_weight,
            original_size_bytes: original_bytes,
            quant_size_bytes: quant_bytes,
            compression_ratio: compression,
            scale,
            zero_point,
            data: packed,
        }
    }
    
    /// Compile a model into METL binary
    pub fn compile(&self, layers: &[Layer]) -> MetlBinary {
        let mut header = MetlHeader::new();
        header.num_layers = layers.len() as u16;
        if let Some(first) = layers.first() { header.input_size = first.input_channels as u32; }
        if let Some(last) = layers.last() { header.output_size = last.output_channels as u32; }
        header.quant_mode = self.quant_mode as u8;
        
        let mut metl_layers = vec![];
        let mut weight_data = vec![];
        let mut bias_data = vec![];
        let mut weight_offset = 0u32;
        let mut bias_offset = 0u32;
        let mut total_cycles = 0u64;
        
        for layer in layers {
            let quant = self.quantize_layer(layer);
            let w_size = quant.data.len() as u32;
            let b_size = (layer.biases.len() * 2) as u32; // INT16 biases
            
            metl_layers.push(MetlLayer {
                layer_id: layer.id as u16,
                op_code: match layer.kernel_size {
                    Some(_) => 1, // Conv
                    None => 0,    // Dense
                },
                input_dim: layer.input_channels as u32,
                output_dim: layer.output_channels as u32,
                weight_offset,
                weight_size: w_size,
                bias_offset,
                activation: match layer.activation {
                    Activation::Relu => 1, Activation::Gelu => 2,
                    Activation::Sigmoid => 3, Activation::Tanh => 4,
                    Activation::None => 0,
                },
                cycles: (layer.weight_count() as f64 * 1.5) as u32,
            });
            
            weight_data.extend_from_slice(&quant.data);
            // INT16 biases
            for &b in &layer.biases {
                let qb = (b / quant.scale).round().clamp(-32768.0, 32767.0) as i16;
                bias_data.extend_from_slice(&qb.to_le_bytes());
            }
            
            weight_offset += w_size;
            bias_offset += b_size;
            total_cycles += metl_layers.last().unwrap().cycles as u64;
        }
        
        // Simple checksum
        let mut all_data = header.to_bytes();
        for ml in &metl_layers { all_data.extend(ml.to_bytes()); }
        all_data.extend(&weight_data);
        all_data.extend(&bias_data);
        let checksum = all_data.iter().fold(0u32, |acc, &b| acc.wrapping_add(b as u32));
        
        header.checksum = checksum;
        
        MetlBinary {
            header, layers: metl_layers,
            weight_data, bias_data,
            total_cycles,
            total_size: all_data.len(),
        }
    }
    
    /// Generate layer execution schedule
    pub fn schedule(&self, layers: &[Layer], bram_size_kb: usize) -> LayerSchedule {
        let mut scheduled = vec![];
        let mut current_cycle = 0u64;
        let mut current_memory = 0usize;
        let mut peak_memory = 0usize;
        
        for (i, layer) in layers.iter().enumerate() {
            let layer_mem = layer.weight_count() + layer.output_channels;
            
            // Check if we need to free memory
            if current_memory + layer_mem > bram_size_kb * 1024 {
                // Unload oldest layers that are no longer needed
                let layers_to_unload = scheduled.iter()
                    .filter(|s| !s.can_pipeline && s.layer_id + 2 < i)
                    .take(2);
                for unload in layers_to_unload {
                    let unload_layer = &layers[unload.layer_id];
                    current_memory = current_memory.saturating_sub(unload_layer.weight_count() + unload_layer.output_channels);
                }
            }
            
            let can_pipeline = i > 0 && layers[i-1].output_channels == layer.input_channels;
            let cycles = (layer.weight_count() as f64 * if can_pipeline { 1.0 } else { 1.5 }) as u64;
            
            scheduled.push(ScheduledLayer {
                layer_id: i,
                start_cycle: current_cycle,
                end_cycle: current_cycle + cycles,
                memory_at_start: current_memory,
                can_pipeline,
            });
            
            current_memory += layer_mem;
            peak_memory = peak_memory.max(current_memory);
            
            if can_pipeline {
                current_cycle += cycles / 2; // Pipeline overlap
            } else {
                current_cycle += cycles;
            }
        }
        
        LayerSchedule {
            layers: scheduled,
            total_cycles: current_cycle,
            peak_memory_bytes: peak_memory * 4, // 4 bytes per param
            pipeline_depth: scheduled.iter().filter(|s| s.can_pipeline).count(),
        }
    }
}

/// A compiled METL binary
#[derive(Debug, Clone)]
pub struct MetlBinary {
    pub header: MetlHeader,
    pub layers: Vec<MetlLayer>,
    pub weight_data: Vec<u8>,
    pub bias_data: Vec<u8>,
    pub total_cycles: u64,
    pub total_size: usize,
}

impl MetlBinary {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = self.header.to_bytes();
        for layer in &self.layers { buf.extend(layer.to_bytes()); }
        buf.extend_from_slice(&self.weight_data);
        buf.extend_from_slice(&self.bias_data);
        buf
    }
    
    /// Calculate compression ratio vs FP32
    pub fn compression_ratio(&self) -> f64 {
        let fp32_size: usize = self.layers.iter()
            .map(|l| (l.input_dim as usize * l.output_dim as usize * 4 + l.output_dim as usize * 4))
            .sum();
        let metl_size = self.weight_data.len() + self.bias_data.len();
        fp32_size as f64 / metl_size.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_layer() -> Layer {
        Layer {
            id: 0, name: "fc1".to_string(),
            input_channels: 64, output_channels: 32,
            kernel_size: None,
            weights: (0..2048).map(|i| (i % 100) as f32 / 100.0).collect(),
            biases: (0..32).map(|i| i as f32 * 0.01).collect(),
            quant_mode: QuantMode::INT8,
            activation: Activation::Relu,
            cycles_estimate: 0, memory_bytes: 0,
        }
    }

    #[test]
    fn test_quantize_int8() {
        let compiler = NeuralCompiler::new(QuantMode::INT8);
        let layer = test_layer();
        let qt = compiler.quantize_layer(&layer);
        assert_eq!(qt.quant_bits, 8);
        assert!(qt.compression_ratio > 1.0);
        assert!(qt.quant_size_bytes < qt.original_size_bytes);
    }

    #[test]
    fn test_quantize_int4() {
        let compiler = NeuralCompiler::new(QuantMode::INT4);
        let layer = test_layer();
        let qt = compiler.quantize_layer(&layer);
        assert_eq!(qt.quant_bits, 4);
        assert!(qt.compression_ratio > 1.0);
    }

    #[test]
    fn test_mixed_quantization() {
        let compiler = NeuralCompiler::new(QuantMode::Mixed);
        let small = Layer {
            id: 0, name: "small".to_string(),
            input_channels: 8, output_channels: 4, kernel_size: None,
            weights: vec![1.0; 32], biases: vec![0.0; 4],
            quant_mode: QuantMode::Mixed, activation: Activation::None,
            cycles_estimate: 0, memory_bytes: 0,
        };
        let big = Layer {
            id: 1, name: "big".to_string(),
            input_channels: 64, output_channels: 64, kernel_size: None,
            weights: vec![1.0; 4096], biases: vec![0.0; 64],
            quant_mode: QuantMode::Mixed, activation: Activation::Relu,
            cycles_estimate: 0, memory_bytes: 0,
        };
        let small_q = compiler.quantize_layer(&small);
        let big_q = compiler.quantize_layer(&big);
        assert_eq!(small_q.quant_bits, 8); // <1K weights → INT8
        assert_eq!(big_q.quant_bits, 4);   // >1K weights → INT4
    }

    #[test]
    fn test_compile_binary() {
        let compiler = NeuralCompiler::new(QuantMode::INT8);
        let layer = test_layer();
        let binary = compiler.compile(&[layer]);
        assert_eq!(binary.layers.len(), 1);
        assert!(binary.total_size > 0);
        assert!(binary.compression_ratio() > 1.0);
    }

    #[test]
    fn test_metl_header() {
        let header = MetlHeader::new();
        let bytes = header.to_bytes();
        assert_eq!(&bytes[0..4], &[b'M', b'E', b'T', b'L']);
    }

    #[test]
    fn test_metl_layer() {
        let layer = MetlLayer {
            layer_id: 0, op_code: 0, input_dim: 64, output_dim: 32,
            weight_offset: 0, weight_size: 1024, bias_offset: 1024,
            activation: 1, cycles: 3000,
        };
        let bytes = layer.to_bytes();
        assert_eq!(bytes.len(), 24);
    }

    #[test]
    fn test_schedule() {
        let compiler = NeuralCompiler::new(QuantMode::INT8);
        let layers = vec![
            test_layer(),
            Layer {
                id: 1, name: "fc2".to_string(),
                input_channels: 32, output_channels: 16, kernel_size: None,
                weights: vec![1.0; 512], biases: vec![0.0; 16],
                quant_mode: QuantMode::INT8, activation: Activation::None,
                cycles_estimate: 0, memory_bytes: 0,
            },
        ];
        let schedule = compiler.schedule(&layers, 256);
        assert_eq!(schedule.layers.len(), 2);
        assert!(schedule.pipeline_depth >= 1); // Can pipeline since 32->32 matches
        assert!(schedule.total_cycles > 0);
    }

    #[test]
    fn test_binary_serialization() {
        let compiler = NeuralCompiler::new(QuantMode::INT8);
        let layer = test_layer();
        let binary = compiler.compile(&[layer]);
        let bytes = binary.to_bytes();
        assert!(bytes.len() > 0);
        assert_eq!(&bytes[0..4], &[b'M', b'E', b'T', b'L']);
    }

    #[test]
    fn test_fp16_quantization() {
        let compiler = NeuralCompiler::new(QuantMode::FP16);
        let layer = test_layer();
        let qt = compiler.quantize_layer(&layer);
        assert_eq!(qt.quant_bits, 16);
        assert_eq!(qt.quant_size_bytes, qt.original_size_bytes); // Same size
    }
}
