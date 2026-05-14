import type { BadgeKey } from "./badges";

export type Publication = {
  id: string;
  title: string;
  authors: string;
  venue: string;
  keywords: string[];
  summary: string;
  paper?: string;
  code?: string;
  badge: BadgeKey;
};

export const publications: Publication[] = [
  {
    id: "foundry",
    title: "Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start",
    authors: "X Liu*, Y Wu*, Y Yao, D Zhuo, I Stoica, ZM Mao",
    venue: "In submission",
    keywords: ["LLM Serving", "CUDA Graphs", "Cold Start", "Context Materialization"],
    summary: "Persists CUDA graph topology and execution context offline, then reconstructs executable graphs online with negligible overhead.",
    paper: "https://arxiv.org/abs/2604.06664",
    code: "https://github.com/foundry-org/foundry",
    badge: "foundry"
  },
  {
    id: "rlboost",
    title: "RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs",
    authors: "Y Wu*, X Liu*, H Zheng, J Gu, B Chen, ZM Mao, A Krishnamurthy, I Stoica",
    venue: "NSDI 2026",
    keywords: ["LLM RL", "Spot Instances", "Kubernetes", "Cost Efficiency"],
    summary: "Offloads rollout workloads to fragmented preemptible resources to reduce LLM RL cost and improve utilization.",
    paper: "https://arxiv.org/abs/2510.19225",
    code: "https://github.com/Terra-Flux/PolyRL",
    badge: "rlboost"
  },
  {
    id: "hetermoe",
    title: "HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs",
    authors: "Y Wu*, X Liu*, S Jin, C Xu, F Qian, ZM Mao, M Lentz, D Zhuo, I Stoica",
    venue: "In submission",
    keywords: ["LLM Training", "MoE", "Heterogeneous GPUs", "DeepSpeed"],
    summary: "Assigns MoE components across mixed GPU generations with zebra parallelism and asymmetric expert placement.",
    paper: "https://arxiv.org/abs/2504.03871",
    badge: "hetermoe"
  },
  {
    id: "plato",
    title: "Plato: Plan to Efficiently Decode for Large Language Model Inference",
    authors: "S Jin*, X Liu*, Y Wu, H Zheng, Q Zhang, M Lentz, ZM Mao, A Prakash, F Qian, D Zhuo",
    venue: "COLM 2025",
    keywords: ["LLM Inference", "Parallel Decoding", "Structured Decoding", "KV Cache"],
    summary: "Decomposes complex queries into dependency graphs to accelerate generation through context-aware parallel decoding.",
    paper: "https://arxiv.org/abs/2402.12280",
    badge: "plato"
  },
  {
    id: "cake",
    title: "Compute Or Load KV Cache? Why Not Both? (CAKE)",
    authors: "S Jin*, X Liu*, Q Zhang, ZM Mao",
    venue: "ICML 2025",
    keywords: ["LLM Inference", "KV Cache", "Long Context", "vLLM", "LMCache"],
    summary: "Reduces long-context prefill latency by overlapping bidirectional KV-cache computation and I/O.",
    paper: "https://arxiv.org/abs/2410.03065",
    badge: "cake"
  },
  {
    id: "lte",
    title: "Learn to be efficient: Build structured sparsity in large language models (LTE)",
    authors: "H Zheng, X Bai, X Liu, ZM Mao, B Chen, F Lai, A Prakash",
    venue: "NeurIPS 2024 Spotlight",
    keywords: ["LLM Efficiency", "Structured Sparsity", "MoE", "Gather-scatter", "Triton"],
    summary: "Trains LLMs to activate fewer neurons while maintaining accuracy, backed by efficient sparse FFN kernels.",
    paper: "https://arxiv.org/abs/2402.06126",
    badge: "lte"
  },
  {
    id: "mm2gb",
    title: "mm2-gb: GPU Accelerated Minimap2 for Long Read DNA Mapping",
    authors: "J Dong*, X Liu*, H Sadasivan, S Sitaraman, S Narayanasamy",
    venue: "ACM BCB 2024 Oral",
    keywords: ["GPU", "DNA Mapping", "Minimap2", "HPC", "Persistent Kernel"],
    summary: "Extends minimap2 with an AMD GPU-accelerated chaining kernel for irregular ultra-long DNA read workloads.",
    paper: "https://www.biorxiv.org/content/10.1101/2024.03.23.586366v2.abstract",
    code: "https://github.com/Minimap2onGPU/mm2-gb",
    badge: "mm2gb"
  }
];
