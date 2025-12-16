---
layout: single
title: "About"
permalink: /About/
header:
  overlay_image: /assets/images/about/selfie-wide.png
comments: false
author_profile: true
---

My name is Xueshen Liu (刘学深). I am a 4th-year Ph.D. candidate in the Computer Science and Engineering Division at the University of Michigan, advised by Prof. [Z. Morley Mao](https://web.eecs.umich.edu/~zmao/). My research interests focus on **distributed systems** and **parallel computing**. Currently, I am exploring efficient solutions for training, inference, and reinforcement learning (RL) of **large language models (LLMs)** by designing **elastic** and **heterogeneous** systems. 

### News

- **Dec. 2025** – Delighted that [**RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs**](https://arxiv.org/abs/2510.19225) was accepted to NSDI'26 — and wrapped up an amazing Student Researcher internship with the Systems Research \@ Google. See you in Seattle!  
- **Jul. 2025** – Excited that [**Plato: Plan to Efficiently Decode for Large Language Model Inference**](https://arxiv.org/abs/2402.12280) will appear at COLM'25. ~~See you in Montreal!~~  
- **May 2025** – [**Compute Or Load KV Cache? Why Not Both? (CAKE)**](https://arxiv.org/abs/2410.03065) is heading to ICML'25 — very happy to share our work on optimzation of long-context KV caches. ~~See you in Vancouver!~~  
- **May 2025** - Excited to join the Systems Research \@ Google ([SRG](https://techsysinfra.google/research/)) as a Student Researcher in Seattle. I will be working on distributed RL frameworks for LLMs with Juncheng Gu and Arvind Krishnamurthy.
- **Dec. 2024** – Honored that [**Learn-To-be-Efficient (LTE): Build Structured Sparsity in Large Language Models**](https://arxiv.org/abs/2402.06126) was presented as a Spotlight at NeurIPS'24. See you in Vancouver!  
- **Oct. 2024** – [**mm2-gb: GPU Accelerated Minimap2 for Long Read DNA Mapping**](https://www.biorxiv.org/content/10.1101/2024.03.23.586366v2.abstract) was selected as an Oral at ACM BCB'24 — thrilled to share our work with the computational biology community.  
- **Sept. 2024** – Very happy to start as Graduate Student Instructor for CSE 589 (Advanced Computer Networks) at the University of Michigan.  
- **May 2024** – Excited to join the Connected Autonomous Vehicle (CAV) Lab at General Motors as a research intern, working on large-scale latency-tolerant positioning systems.  

### Education

- **University of Michigan**, Ann Arbor, MI  
  Ph.D. in Computer Science and Engineering (CSE), 2022 – Present  
  B.S. in Computer Science and Engineering (CSE), 2020 – 2022
- **Shanghai Jiao Tong University**, Shanghai, China  
  B.S. in Electrical and Computer Engineering (ECE), 2018 – 2022

### Selected Projects & Publications

- **GPU States Checkpointing for Distributed Elastic Serving Systems** (Ongoing, Sept. 2025 – Present)  
  Saving selected ranges of CUDA states into an image to skip warmup during elastic serving.
- [**RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs**](https://arxiv.org/abs/2510.19225) (NSDI'26, May 2025 – Dec. 2025)  
  Designed a rollout system that adaptively offloads LLM RL workloads to preemptible instances, achieving up to 49% cost reduction and improving utilization of fragmented cloud resources. Code available [here](https://github.com/Terra-Flux/PolyRL).
- [**HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs**](https://arxiv.org/abs/2504.03871) (In submission, Apr. 2024 – Apr. 2025)  
  Disaggregates MoE models and assigns experts to older GPUs (e.g., V100, T4) while using newer GPUs for attention, with zebra parallelism and asymmetric expert assignment for fine-grained load balancing.
- [**Plato: Plan to Efficiently Decode for Large Language Model Inference**](https://arxiv.org/abs/2402.12280) (COLM'25, Oct. 2024 – Jul. 2025)  
  Decomposes complex queries into a dependency graph and accelerates generation through context-aware parallel decoding and structured decoding.
- [**Compute Or Load KV Cache? Why Not Both? (CAKE)**](https://arxiv.org/abs/2410.03065) (ICML'25, Sept. 2024 – Feb. 2025)  
  Reduces LLM prefill latency on long-context inputs via bidirectional KV-cache generation that overlaps computation and I/O, built on top of vLLM and LMCache.
- [**Learn-To-be-Efficient (LTE): Build Structured Sparsity in Large Language Models**](https://arxiv.org/abs/2402.06126) (NeurIPS'24 Spotlight, Mar. 2024 – Oct. 2024)  
  Trains LLMs to activate fewer neurons through structured sparsity while maintaining accuracy, with an efficient Triton/CUDA gather-scatter MLP kernel that achieves near-linear speedup with sparsity.
- [**mm2-gb: GPU Accelerated Minimap2 for Long Read DNA Mapping**](https://www.biorxiv.org/content/10.1101/2024.03.23.586366v2.abstract) (ACM BCB'24 Oral, May 2022 – Oct. 2024)  
  Extends minimap2-v2.24 with an AMD GPU-accelerated chaining kernel using HIP and persistent kernels to tackle extremely irregular ultra-long DNA read workloads. Code available [here](https://github.com/Minimap2onGPU/mm2-gb).

### Experience

- **Student Researcher, Systems Research @ Google**, Seattle, WA (May 2025 – Dec. 2025)  
  - Characterized bottlenecks across the LLM RL pipeline and identified rollout as a dominant yet highly elastic component.  
  - Designed RLBoost on Google Cloud Platform to harvest fragmented spot resources, lower RL training cost, and improve overall utilization.  
  - Explored heterogeneous compute options (multi-generation GPUs & TPUs) under diverse RL workloads (sequence length, tool calling, etc.).  
  - Contributed to an NL2SQL agentic training pipeline, optimizing multi-node communication and applying asynchronous tool calling.
- **Graduate Student Instructor, CSE 589 Advanced Computer Networks, University of Michigan**, Ann Arbor, MI (Sept. 2024 – Dec. 2024)  
  - Led in-class discussions and held regular office hours.  
  - Delivered a guest lecture on distributed software-defined networking (dSDN).  
  - Mentored graduate students on research projects, including methodology, implementation, and presentation.  
- **Intern Researcher, Connected Autonomous Vehicle (CAV) Lab, General Motors**, Warren, MI (May 2024 – Aug. 2024)  
  - Designed a large-scale latency-tolerant vehicle positioning system on edge/cloud servers.  
  - Developed a deep factor graph model to handle delayed perception data while maintaining real-time responsiveness.  
  - Leveraged parallelism and prioritized scheduling to meet tight latency constraints.  

### Service & Honors

- **Reviewer / PC member**: ICLR'26, ICLR'25, COLING'25  
- **Invited talk**: "Scalable & Latency-tolerant Edge/Cloud Computing via Deep Factor Graph" (Aug. 2024)  
- **Invited talk**: "Minimap2-gigabases (mm2-gb)" at AMD HPC Apps Knowledge Sync (May 2024)  
- **Awards**:
  - Roger King Scholarship, College of Engineering, University of Michigan (Aug. 2021)
  - Runner-up Team & Grand Prize, 18th Robomaster Final Competition (Aug. 2019)

### Skills

- **Machine learning & systems**: VeRL, PyTorch, DeepSpeed, NCCL, SGLang, vLLM, FlashAttention, LMCache, CUTLASS  
- **Programming languages**: Python, Rust, Triton, CUDA, HIP, C/C++, Go, LLVM  
- **Development & profiling**: Kubernetes, Nsight Systems/Compute, MCP, Cursor/Codex, Perfetto, Slurm, Docker, Git  
