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

## 🚧 **TODO List**

### Upcoming Features
- [ ] **qwen3_30B_A3B.py**: Support for larger Qwen3-30B-A3B model
- [ ] **Multi-GPU Support**: Enhanced tensor parallelism for distributed inference
- [ ] **More Model Variants**: Support for additional Qwen model sizes and configurations
- [ ] **Performance Optimizations**: Further kernel optimizations and memory efficiency improvements

### Current Support
- [x] **qwen3_0_6B.py**: Complete implementation for Qwen3-0.6B model
- [x] **Basic vLLM Features**: PagedAttention, KV caching, continuous batching
- [x] **Flash Attention**: Auto-detection and fallback support

## 🙏 **Acknowledgments**

This project is inspired by and based on the concepts from [vLLM](https://github.com/vllm-project/vllm), a high-throughput and memory-efficient inference and serving engine for LLMs. We are grateful to the vLLM team and community for their pioneering work in LLM inference optimization.

Also based on the excellent [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) project. Thanks to the original authors for their outstanding work! 

## 📚 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=amulil/cleanvllm&type=Date)](https://star-history.com/#amulil/cleanvllm&Date)