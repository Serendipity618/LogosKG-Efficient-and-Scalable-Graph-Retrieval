# LogosKG: Scaling Biomedical Knowledge Graph Retrieval for Interpretable Reasoning

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Backend: PyTorch | Numba | SciPy](https://img.shields.io/badge/backend-PyTorch%20%7C%20Numba%20%7C%20SciPy-orange)

**LogosKG** is a model-agnostic and highly optimized engine for large-scale multi-hop knowledge graph (KG) retrieval. Designed to mitigate the memory bottlenecks and pointer-chasing latency typical of traditional graph libraries, LogosKG provides efficient, hardware-accelerated traversal tailored for **LLM-KG applications** and complex reasoning systems.

### Authors & Affiliations

1. **He Cheng**, University of Colorado Anschutz Medical Campus, Aurora, CO, USA
2. **Yifu Wu**, University of Colorado Anschutz Medical Campus, Aurora, CO, USA
3. **Saksham Khatwani**, University of Colorado Anschutz Medical Campus, Aurora, CO, USA & University of Colorado Boulder, Boulder, CO, USA
4. **Maya Kruse**, University of Colorado Anschutz Medical Campus, Aurora, CO, USA
5. **Dmitriy Dligach**, Loyola University Chicago, Chicago, IL, USA
6. **Timothy A. Miller**, Harvard Medical School, Boston, MA, USA & Boston Children's Hospital, Boston, MA, USA
7. **Majid Afshar**, University of Wisconsin-Madison, Madison, WI, USA
8. **Yanjun Gao***, University of Colorado Anschutz Medical Campus, Aurora, CO, USA

*(Note: * denotes corresponding author)*

---

## 🧠 Core Methodology: Vectorized Topology

Unlike object-oriented graph implementations, LogosKG translates the graph topology into aligned **Compressed Sparse Row (CSR)** matrices. By mapping entities and relations to integer indices, the knowledge graph is represented via three core matrices:

1. **Subject Matrix (`sub_matrix`):** Maps entities to triplet indices.
2. **Object Matrix (`obj_matrix`):** Maps triplet indices to entities.
3. **Relation Matrix (`rel_matrix`):** Maps triplet indices to relation types.

**The Hop Kernel:** A $k$-hop neighbor retrieval is mathematically reduced to sparse Boolean matrix multiplications. Starting with a seed entity frontier $H_0$, the subsequent hop is computed as:

$$
H_{k+1} = H_k \mathbin{@} \mathbf{M}_{\text{sub}} \mathbin{@} \mathbf{M}_{\text{obj}}
$$

This matrix-based formulation allows LogosKG to bypass Python's Global Interpreter Lock (GIL) and achieve highly concurrent execution using **Numba (JIT CPU)** or **PyTorch (GPU batching)**.

## 📊 Data Format

LogosKG readily accepts a standard Python list of tuples representing your knowledge graph. Each tuple represents a directed edge in the format `(head_entity, relation, tail_entity)`. All elements must be strings.

**Example of the internal triplet structure:**
```python
[
    ("C0002871", "has_symptom", "C0011849"),
    ("C0011849", "is_a", "C1234567"),
    ("C1234567", "associated_with", "C0002871"),
]
```
**Data Resource:** Pre-built UMLS SNOMED CUI graph object (featuring physician-selected relations pertinent to clinical diagnosis): [Download (700 MB)](https://drive.google.com/file/d/1zlb0zey_tAnFWtCY_NvhA0dqfydL4Ph7/view?usp=sharing)

> **Reference:** The customized clinical relations and graph subsets are derived from the [DR.KNOWs repository](https://github.com/serenayj/DRKnows?tab=readme-ov-file).

## ⚖️ Architecture Variants

LogosKG provides two distinct architectures depending on hardware constraints and graph scale:

1. **LogosKG (In-Memory Engine):** Designed for graphs that fit within system RAM. It loads the full CSR matrices into memory, delivering maximum throughput and sub-millisecond latency for batch processing.
2. **LogosKGLarge (Out-of-Core Engine):** Designed for massive graphs exceeding physical memory. It utilizes a disk-backed graph partitioner (`utils/KGPartitioner.py`) to slice the graph into cohesive sub-graphs, employing an LRU cache and query reordering algorithms to minimize disk I/O.

## ⚙️ Installation & Project Structure

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Serendipity618/LogosKG-Efficient-and-Scalable-Graph-Retrieval.git
cd LogosKG-Efficient-and-Scalable-Graph-Retrieval
pip install -r requirements.txt
```

**Repository Structure:**
* `LogosKG.py`: Core in-memory retrieval engine.
* `LogosKGLarge.py`: Out-of-core retrieval engine for massive graphs.
* `utils/`: Contains the `KGPartitioner.py` and evaluation baselines (`baselines_cpu.py`, `baselines_gpu.py`).

## 📖 Quick Start

### 1. In-Memory Engine (LogosKG)

```python
import pickle
from LogosKG import LogosKG

# Load your graph data (list of tuples)
with open("SNOMED_CUI_MAJID_Graph_wSelf.pkl", "rb") as f:
    triplet_data = pickle.load(f)

# Initialize the engine (supports 'numba', 'scipy', or 'torch' backend)
kg = LogosKG(triplets=triplet_data, backend="numba")

# Define target entities
seed_entities = ["C0002871", "C0011849"]

# Retrieve entities within k hops
entities = kg.retrieve_within_k_hop(
    entity_ids=seed_entities, 
    hops=2,
    shortest_path=True
)

# Retrieve human-readable paths for interpretability
path_data = kg.retrieve_with_paths_within_k_hop(
    entity_ids=seed_entities, 
    hops=2,
    shortest_path=True,
)
print(path_data["paths"])
```

### 2. Disk-Backed Engine (LogosKGLarge)

The API remains consistent, allowing scalable integration without modifying core application logic.

```python
from LogosKGLarge import LogosKGLarge

kg_large = LogosKGLarge(
    partition_dir="./graph_partitions",
    triplets="massive_knowledge_graph.pkl", 
    num_partitions=16,
    cache_size=5,     
    backend="torch",  
    device="cuda:0"
)

results = kg_large.retrieve_within_k_hop(entity_ids=["C1234567"], hops=3)
```

## 🎓 Citation

If you find LogosKG useful in your research, please cite our paper:

```bibtex
@article{cheng2026scaling,
  title={Scaling Biomedical Knowledge Graph Retrieval for Interpretable Reasoning: Applications to Clinical Diagnosis Prediction},
  author={Cheng, He and Wu, Yifu and Khatwani, Saksham and Kruse, Maya and Dligach, Dmitriy and Miller, Timothy A. and Afshar, Majid and Gao, Yanjun},
  journal={medRxiv},
  pages={2026--01},
  year={2026},
  publisher={Cold Spring Harbor Laboratory Press},
  doi={10.64898/2026.01.12.26343957},
  url={[https://www.medrxiv.org/content/10.64898/2026.01.12.26343957v1](https://www.medrxiv.org/content/10.64898/2026.01.12.26343957v1)}
}
```

## 📄 License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


