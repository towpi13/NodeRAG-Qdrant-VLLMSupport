# NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes
 
<div align="center">
  <img src="/asset/Node_background.jpg" alt="NodeRAG Logo" width="600px">
  
  <p>
    <a href="https://arxiv.org/abs/arkiv"><img src="https://img.shields.io/badge/arXiv-arkiv-b31b1b.svg" alt="arXiv"></a>
    <a href="https://pypi.org/project/NodeRAG/"><img src="https://img.shields.io/pypi/v/NodeRAG.svg" alt="PyPI"></a>
    <a href="https://github.com/Terry-Xu-666/NodeRAG/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <a href="https://github.com/Terry-Xu-666/NodeRAG/issues"><img src="https://img.shields.io/github/issues/Terry-Xu-666/NodeRAG.svg" alt="GitHub issues"></a>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"></a>
    <a href="https://terry-xu-666.github.io/NodeRAG_web/"><img src="https://img.shields.io/badge/Website-NodeRAG-green" alt="Website"></a>
    <a href="https://github.com/Terry-Xu-666/NodeRAG"><img src="https://img.shields.io/github/stars/Terry-Xu-666/NodeRAG.svg?style=social" alt="GitHub stars"></a>
  </p>
</div>

## 📢 News

- **[2025-03-18]** 🚀 **NodeRAG v0.1.0 Released!** The first stable version is now available on [PyPI](https://pypi.org/project/NodeRAG/). Install it with `pip install NodeRAG`.
  
- **[2025-03-18]** 🌐 **Official Website Launched!** Visit [NodeRAG_web](https://terry-xu-666.github.io/NodeRAG_web/) for comprehensive documentation, tutorials, and examples.

---

🚀 NodeRAG is a heterogeneous graph-based generation and retrieval RAG system that you can install and use in multiple ways. 🖥️ We also provide a user interface (local deployment) and convenient tools for visualization generation. You can read our [paper](#) 📄 to learn more. For experimental discussions, check out our [blog posts](https://terry-xu-666.github.io/NodeRAG_web/blog/) 📝. 

---

## 🚀 Quick Start

📖 View our official website for comprehensive documentation and tutorials:  
👉 [NodeRAG_web](https://terry-xu-666.github.io/NodeRAG_web/)

### 🧩 Workflow

<div align="center">
  <img src="/asset/NodeGraph_Figure2.png" alt="NodeRAG Workflow" width="800px">
</div>

---

## NodeRAG

### Conda Setup

Create and activate a virtual environment for NodeRAG:

```bash
conda create -n NodeRAG python=3.10
conda activate NodeRAG
```

---

### Install `uv` (Optional: Faster Package Installation)

To speed up package installation, use [`uv`](https://github.com/astral-sh/uv):

```bash
pip install uv
```

---

### Install NodeRAG

Install NodeRAG using `uv` for optimized performance:

```bash
uv pip install NodeRAG
```

### Next
> For indexing and answering processes, please refer to our website: [Indexing](https://terry-xu-666.github.io/NodeRAG_web/docs/indexing/) and [Answering](https://terry-xu-666.github.io/NodeRAG_web/docs/answer/)


## ✨ Features



#### 🔗 Enhancing Graph Structure for RAG  
NodeRAG introduces a heterogeneous graph structure that strengthens the foundation of graph-based Retrieval-Augmented Generation (RAG).



#### 🔍 Fine-Grained and Explainable Retrieval  
NodeRAG leverages HeteroGraphs to enable functionally distinct nodes, ensuring precise and context-aware retrieval while improving interpretability.

#### 🧱 A Unified Information Retrieval  
Instead of treating extracted insights and raw data as separate layers, NodeRAG integrates them as interconnected nodes, creating a seamless and adaptable retrieval system.


#### ⚡ Optimized Performance and Speed  
NodeRAG achieves faster graph construction and retrieval speeds through unified algorithms and optimized implementations.


#### 🔄 Incremental Graph Updates  
NodeRAG supports incremental updates within heterogeneous graphs using graph connectivity mechanisms.



#### 📊 Visualization and User Interface  
NodeRAG offers a user-friendly visualization system. Coupled with a fully developed Web UI, users can explore, analyze, and manage the graph structure with ease.

## ⚙️ Performance

### 📊 Benchmark Performance

<div align="center">
  <img src="/asset/performance.png" alt="Benchmark Performance" width="800px">
</div>

*NodeRAG demonstrates strong performance across multiple benchmark tasks, showcasing efficiency and retrieval quality.*

---

### 🖥️ System Performance

<div align="center">
  <img src="/asset/system_performance.png" alt="System Performance" width="800px">
</div>

*Optimized for speed and scalability, NodeRAG achieves fast indexing and query response times even on large datasets.*