import type { BadgeKey } from "./badges";

export type Lane = "education" | "work" | "publications" | "blogs";
export type MarkerKind = "start" | "end" | "release" | "acceptance" | "job" | "post" | "award" | "talk";

export type LaneTraceEvent = {
  id: string;
  date: string;
  value: number;
  label: string;
  detail: string;
  duration?: string;
  kind: MarkerKind | "service" | "education" | "paper" | "blog";
  logo?: BadgeKey;
  link?: string;
  trace?: boolean;
};

export type LaneTrace = {
  lane: Lane;
  title: string;
  subtitle: string;
  color: string;
  unit: string;
  events: LaneTraceEvent[];
};

export const laneTraces: LaneTrace[] = [
  {
    lane: "education",
    title: "GPU0 Education",
    subtitle: "academic milestones",
    color: "#d8d8d8",
    unit: "academic context",
    events: [
      { id: "sjtu-start", date: "2018-09", value: 55, label: "SJTU B.S. ECE", detail: "Started B.S. in Electrical and Computer Engineering at Shanghai Jiao Tong University.", duration: "Sep 2018-May 2022", kind: "start", logo: "sjtu" },
      { id: "robomaster", date: "2019-08", value: 55, label: "Robomaster Final Competition", detail: "Runner-up Team and Grand Prize.", duration: "Aug 2019", kind: "award", logo: "sjtu", trace: false },
      { id: "umich-bs", date: "2020-09", value: 72, label: "University of Michigan B.S. CSE", detail: "Started B.S. in Computer Science and Engineering at the University of Michigan.", duration: "Sep 2020-May 2022", kind: "start", logo: "umich" },
      { id: "roger-king", date: "2021-08", value: 72, label: "Roger King Scholarship", detail: "College of Engineering, University of Michigan.", duration: "Aug 2021", kind: "award", logo: "umich", trace: false },
      { id: "bs-complete", date: "2022-05", value: 18, label: "Completed B.S. degrees", detail: "Completed undergraduate study before starting the Ph.D. program.", duration: "May 2022", kind: "education", logo: "umich" },
      { id: "umich-phd", date: "2022-09", value: 88, label: "University of Michigan Ph.D. CSE", detail: "Started Ph.D. in Computer Science and Engineering, advised by Prof. Z. Morley Mao.", duration: "Sep 2022-present", kind: "start", logo: "umich" },
      { id: "cse589", date: "2024-09", value: 88, label: "Graduate Student Instructor for CSE 589", detail: "Advanced Computer Networks at University of Michigan.", duration: "Fall 2024", kind: "education", logo: "umich", trace: false },
      // { id: "pc-service", date: "2025-01", value: 88, label: "Reviewer / PC service", detail: "ICLR'26, ICLR'25, and COLING'25.", kind: "service", logo: "neurips", trace: false }
    ]
  },
  {
    lane: "work",
    title: "GPU1 Work",
    subtitle: "internships and industry runtime",
    color: "#c6c6c6",
    unit: "engagement",
    events: [
      { id: "gm-start", date: "2024-05", value: 70, label: "General Motors CAV Lab", detail: "Research intern on latency-tolerant vehicle positioning systems.", duration: "May-Aug 2024", kind: "job", logo: "gm" },
      { id: "gm-end", date: "2024-08", value: 0, label: "General Motors CAV Lab", detail: "End of my internship at General Motors CAV Lab.", duration: "May-Aug 2024", kind: "end", logo: "gm" },
      { id: "google-fulltime", date: "2025-05", value: 96, label: "Google Systems Research", detail: "Full-time summer student researcher on distributed RL frameworks for LLMs.", duration: "May-Aug 2025", kind: "job", logo: "google" },
      { id: "google-parttime", date: "2025-09", value: 48, label: "Google Systems Research part-time", detail: "Continued part-time work during the academic term.", duration: "Sep 2025-Mar 2026", kind: "job", logo: "google" },
      { id: "google-end", date: "2026-03", value: 0, label: "Google Systems Research part-time", detail: "End of my part-time work during the academic term.", duration: "Sep 2025-Mar 2026", kind: "end", logo: "google" },
      { id: "citadel-start", date: "2026-06", value: 98, label: "Citadel Securities internship starts", detail: "Incoming full-time quantitative researcher internship.", duration: "Summer 2026", kind: "job", logo: "citadel" }
    ]
  },
  {
    lane: "publications",
    title: "GPU2 Publications",
    subtitle: "research and projects",
    color: "#b4b4b4",
    unit: "research output",
    events: [
      { id: "mm2gb-start", date: "2022-05", value: 28, label: "mm2-gb", detail: "Started GPU-accelerated minimap2 work for long-read DNA mapping.", duration: "May 2022-Nov 2024", kind: "start", logo: "mm2gb", link: "https://www.biorxiv.org/content/10.1101/2024.03.23.586366v2.abstract" },
      { id: "plato-start", date: "2024-02", value: 50, label: "Plato", detail: "Started parallel decoding with dependency-aware planning.", duration: "Feb 2024-Aug 2025", kind: "start", logo: "plato", link: "https://arxiv.org/abs/2402.12280" },
      { id: "lte-start", date: "2024-03", value: 72, label: "LTE", detail: "Started structured sparsity and efficient sparse FFN kernel work.", duration: "Mar 2024-Jan 2025", kind: "start", logo: "lte", link: "https://arxiv.org/abs/2402.06126" },
      { id: "mm2gb-bcb", date: "2024-10", value: 72, label: "mm2-gb accepted to ACM BCB'24 Oral", detail: "", duration: "ACM BCB 2024", kind: "acceptance", logo: "bcb", link: "https://www.biorxiv.org/content/10.1101/2024.03.23.586366v2.abstract", trace: false },
      { id: "cake-start", date: "2024-10", value: 82, label: "CAKE", detail: "Started bidirectional KV-cache generation for long-context LLM inference.", duration: "Oct 2024-Jun 2025", kind: "start", logo: "cake", link: "https://arxiv.org/abs/2410.03065" },
      { id: "mm2gb-end", date: "2024-11", value: 64, label: "mm2-gb active work ends", detail: "Main mm2-gb active project phase completed after BCB acceptance.", duration: "May 2022-Nov 2024", kind: "end", logo: "mm2gb" },
      { id: "lte-neurips", date: "2024-12", value: 64, label: "LTE accepted to NeurIPS'24 Spotlight", detail: "", duration: "NeurIPS 2024", kind: "acceptance", logo: "neurips", link: "https://arxiv.org/abs/2402.06126", trace: false },
      { id: "lte-end", date: "2025-01", value: 44, label: "LTE active work ends", detail: "Main LTE active project phase completed after NeurIPS.", duration: "Mar 2024-Jan 2025", kind: "end", logo: "lte" },
      { id: "hetermoe-start", date: "2025-01", value: 76, label: "HeterMoE", detail: "Started efficient MoE training on heterogeneous GPUs.", duration: "Jan 2025-present", kind: "start", logo: "hetermoe", link: "https://arxiv.org/abs/2504.03871" },
      { id: "rlboost-start", date: "2025-05", value: 88, label: "RLBoost", detail: "Started preemptible-resource systems work for cost-efficient LLM RL.", duration: "May 2025-Jan 2026", kind: "start", logo: "rlboost", link: "https://arxiv.org/abs/2510.19225" },
      { id: "cake-icml", date: "2025-05", value: 88, label: "CAKE accepted to ICML'25", detail: "", duration: "ICML 2025", kind: "acceptance", logo: "icml", link: "https://arxiv.org/abs/2410.03065", trace: false },
      { id: "cake-end", date: "2025-06", value: 70, label: "CAKE active work ends", detail: "Main CAKE active project phase completed after ICML acceptance.", duration: "Oct 2024-Jun 2025", kind: "end", logo: "cake" },
      { id: "plato-colm", date: "2025-07", value: 88, label: "Plato accepted to COLM'25", detail: "", duration: "COLM 2025", kind: "acceptance", logo: "colm", link: "https://arxiv.org/abs/2402.12280", trace: false },
      { id: "plato-end", date: "2025-08", value: 62, label: "Plato active work ends", detail: "Main Plato active project phase completed after COLM acceptance.", duration: "Feb 2024-Aug 2025", kind: "end", logo: "plato" },
      { id: "foundry-start", date: "2025-09", value: 96, label: "Foundry", detail: "Started CUDA graph context materialization work for fast LLM serving cold start.", duration: "Sep 2025-present", kind: "start", logo: "foundry", link: "https://arxiv.org/abs/2604.06664" },
      { id: "rlboost-nsdi", date: "2025-12", value: 96, label: "RLBoost accepted to NSDI'26", detail: "", duration: "NSDI 2026", kind: "acceptance", logo: "nsdi", link: "https://arxiv.org/abs/2510.19225", trace: false },
      { id: "rlboost-end", date: "2026-01", value: 78, label: "RLBoost active work ends", detail: "Main RLBoost active project phase completed after NSDI acceptance.", duration: "May 2025-Jan 2026", kind: "end", logo: "rlboost" },
    ]
  },
  {
    lane: "blogs",
    title: "GPU3 Blogs",
    subtitle: "sharing my learnings",
    color: "#9a9a9a",
    unit: "posts",
    events: [
      { id: "cuda-uvm-profile", date: "2024-03", value: 1, label: "Profile CUDA UVM Performance", detail: "First CUDA UVM profiling note.", duration: "single post", kind: "post", logo: "cuda", link: "/posts/cuda_uvm_profile/" },
      { id: "cuda-uvm", date: "2024-03", value: 2, label: "Understand CUDA Unified Memory", detail: "CUDA UVM experiment log.", duration: "single post", kind: "post", logo: "cuda", link: "/posts/cuda_uvm/" },
      { id: "triton-gather", date: "2024-04", value: 3, label: "Gather-scatter GEMM with Triton", detail: "Triton gather/scatter matrix multiplication kernel.", duration: "single post", kind: "post", logo: "triton", link: "/posts/triton_gather_scatter/" },
      { id: "cuda-graph", date: "2024-05", value: 4, label: "CUDA Graph and StaticCache", detail: "LLM inference prototype with CUDA graph.", duration: "single post", kind: "post", logo: "cuda", link: "/posts/cuda_graph/" },
      { id: "cutlass", date: "2024-06", value: 5, label: "Custom Gather-scatter Operator by CUTLASS", detail: "Custom operator implementation notes.", duration: "single post", kind: "post", logo: "cuda", link: "/posts/cutlass/" },
      { id: "deepspeed-moe", date: "2024-06", value: 6, label: "Training Custom Mixtral Model with DeepSpeed", detail: "DeepSpeed MoE training placeholder note.", duration: "single post", kind: "post", logo: "pytorch", link: "/posts/deepspeed_moe/" },
      { id: "deepspeed-profile", date: "2024-11", value: 7, label: "Nsight profiling with DeepSpeed", detail: "Profiling multi-node DeepSpeed training on a cluster.", duration: "single post", kind: "post", logo: "nccl", link: "/posts/deepspeed_profile/" },
      { id: "triton-ffn", date: "2024-12", value: 8, label: "Gather-scatter FFN kernel with Triton", detail: "Efficient sparse FFN kernel for structured sparsity.", duration: "single post", kind: "post", logo: "triton", link: "/posts/triton_gather_scatter_ffn/" },
      { id: "claude-tmux", date: "2026-02", value: 9, label: "Claude Code Docker sandbox", detail: "Docker/tmux sandbox for long-running coding agents.", duration: "single post", kind: "post", logo: "claude", link: "/posts/claude_tmux/" }
    ]
  }
];
