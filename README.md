# CleanvLLM

**A single-file educational implementation for understanding vLLM's core concepts and running LLM inference.**

## 🎯 **Purpose - Why This Matters**

**Learn AI Infrastructure Fundamentals**: This project provides a clean, educational implementation of vLLM's core concepts in a single Python file, making it easy to understand how modern LLM inference engines work under the hood.

**Perfect for Learning**: Whether you're a student, researcher, or engineer wanting to understand vLLM internals, this simplified implementation helps you grasp the fundamental concepts without getting lost in production complexity.

## 🚀 **Quick Start - Run vLLM Inference in 3 Steps**

```bash
# 1. Create and activate conda environment
conda create -n cleanvllm python=3.10 -y && conda activate cleanvllm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run vLLM inference
python qwen3_0_6B.py
```

**That's it! You're now running vLLM inference!**

## 📖 **Detailed Usage**

### Basic Usage

1. Update the model path in `qwen3_0_6B.py`:
```python
path = os.path.expanduser("~/path/to/your/qwen3model")
```

2. Run the script:
```bash
python qwen3_0_6B.py
```

### Configuration Options
- **Model Path**: Update the `path` variable in `qwen3_0_6B.py` to point to your model directory
- **GPU Memory**: Adjust `gpu_memory_utilization` parameter based on your GPU memory capacity

## 🚧 **TODO List**

### Upcoming Features
- [ ] **qwen3_30B_A3B.py**: Support for larger Qwen3-30B-A3B model
- [ ] **Multi-GPU Support**: Enhanced tensor parallelism for distributed inference
- [ ] **More Model Variants**: Support for additional Qwen model sizes and configurations
- [ ] **Performance Optimizations**: Further kernel optimizations and memory efficiency improvements

### Current Support
- ✅ **qwen3_0_6B.py**: Complete implementation for Qwen3-0.6B model
- ✅ **Basic vLLM Features**: PagedAttention, KV caching, continuous batching
- ✅ **Flash Attention**: Auto-detection and fallback support

## 🙏 **Acknowledgments**

This project is inspired by and based on the concepts from [vLLM](https://github.com/vllm-project/vllm), a high-throughput and memory-efficient inference and serving engine for LLMs. We are grateful to the vLLM team and community for their pioneering work in LLM inference optimization.

Also based on the excellent [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) project. Thanks to the original authors for their outstanding work! 

## 📊 **Sample Output**

When you run the script, you'll see output similar to this:

```
Flash Attention not available, using PyTorch native attention
Environment Check:
   PyTorch Version: 2.7.1+cu118
   CUDA Available: True
   GPU Count: 1
   Current GPU: 0
   Flash Attention: Not Available
Initializing Qwen3-0.6B model...
GPU Device Mapping: CUDA Device 0 -> Physical GPU 1
GPU Memory: Total 23.0GB, Used 2.0GB, Free 21.0GB
KV Cache Allocation: 79 blocks, 28.0MB per block, 2212.0MB total
Model initialization complete!
Applying chat template...
Starting text generation...
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.67s/it, Prefill=8tok/s, Decode=27tok/s]

================================================================================
Generation Results:
================================================================================


Prompt: '<|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant\n'
Completion: '<think>\nOkay, the user asked, "Who are you?" I need to respond in a helpful and friendly way. Let me start by acknowledging their question. I should mention that I\'m an AI assistant, but also add something about my purpose. I should make sure my answer is not just a simple statement, but rather a bit more engaging. Maybe I can mention that I\'m here to help with questions or provide information. Let me check if there\'s a specific context I should consider. Since the user just asked who I am, I should keep it general but still convey that I\'m here to assist. I should avoid any technical jargon. Let me make sure the response is natural and not too formal.\n</think>\n\nI\'m an AI assistant designed to help with questions and provide information. If you have any questions or need assistance, feel free to ask! 😊<|im_end|>'
----------------------------------------

Generation complete!

Usage Instructions:
- This script integrates complete LLM inference functionality
- Supports paged KV Cache management
- Supports prefix caching optimization
- Supports continuous batching
- Supports tensor parallelism (requires multiple GPUs)
- Auto-adapts to Flash Attention (if available) or PyTorch native attention
- Modify the path variable to point to your Qwen3-0.6B model directory

Configuration:
- enforce_eager=True: Disable CUDA graph optimization, suitable for debugging
- gpu_memory_utilization=0.10: Use 10% of GPU memory
- temperature=0.6: Sampling temperature
- max_tokens=256: Maximum generation tokens

Performance Optimization:
- Install Flash Attention for best performance: pip install flash-attn
- If Flash Attention is installed, the script automatically uses optimized implementation
- If not installed, the script uses compatible PyTorch native implementation
```

This output shows the complete inference process including environment setup, model loading, and text generation with performance metrics. 