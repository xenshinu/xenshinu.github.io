import type { BadgeKey } from "./badges";
import { publications } from "./publications";

export type Lane = "education" | "work" | "publications" | "blogs";
export type MarkerKind = "start" | "end" | "release" | "acceptance" | "job" | "post" | "award" | "talk";

export type LaneTraceEvent = {
  id: string;
  date: string;
  value: number;
  label: string;
  detail: string;
  details?: string[];
  duration?: string;
  kind: MarkerKind | "service" | "education" | "paper" | "blog";
  logo?: BadgeKey;
  link?: string;
  trace?: boolean;
  process?: boolean;
};

export type LaneTrace = {
  lane: Lane;
  title: string;
  subtitle: string;
  color: string;
  unit: string;
  events: LaneTraceEvent[];
};

function publicationRefs(id: string): Pick<LaneTraceEvent, "label" | "detail" | "details" | "link"> {
  const publication = publications.find((item) => item.id === id);

  return {
    label: publication?.title ?? "",
    detail: publication?.summary ?? "",
    details: publication?.details,
    link: publication?.paper
  };
}

export const laneTraces: LaneTrace[] = [
  {
    lane: "education",
    title: "GPU0 Education",
    subtitle: "academic milestones",
    color: "#d8d8d8",
    unit: "depth",
    events: [
      {
        id: "sjtu-start",
        date: "2018-09",
        value: 55,
        label: "SJTU B.S. ECE",
        detail: "B.S. in Electrical and Computer Engineering at Shanghai Jiao Tong University.",
        duration: "Sep 2018-Aug 2022",
        kind: "education",
        logo: "sjtu"
      },
      {
        id: "umich-bs",
        date: "2020-09",
        value: 72,
        label: "University of Michigan B.S. CSE",
        detail: "B.S. in Computer Science and Engineering at the University of Michigan.",
        duration: "Sep 2020-May 2022",
        kind: "education",
        logo: "umich"
      },
      {
        id: "umich-end",
        date: "2022-05",
        value: 55,
        label: "UMICH CS B.S. degrees completed",
        detail: "",
        duration: "May 2022",
        kind: "end",
        logo: "umich"
      },
      {
        id: "sjtu-end",
        date: "2022-08",
        value: 18,
        label: "SJTU ECE B.S. degrees completed",
        detail: "",
        duration: "Aug 2022",
        kind: "end",
        logo: "sjtu"
      },
      {
        id: "umich-phd",
        date: "2022-08",
        value: 88,
        label: "University of Michigan Ph.D. CSE",
        detail: "Ph.D. in Computer Science and Engineering at the University of Michigan, advised by Prof. Z. Morley Mao.",
        duration: "Aug 2022-present",
        kind: "education",
        logo: "umich"
      },
      {
        id: "iclr25-reviewer",
        date: "2024-10",
        value: 88,
        label: "ICLR'25 reviewer",
        detail: "Reviewer service for ICLR 2025.",
        duration: "Oct 2024",
        kind: "service",
        logo: "iclr",
        trace: false,
        process: false
      },
      {
        id: "iclr26-reviewer",
        date: "2025-10",
        value: 88,
        label: "ICLR'26 reviewer",
        detail: "Reviewer service for ICLR 2026.",
        duration: "Oct 2025",
        kind: "service",
        logo: "iclr",
        trace: false,
        process: false
      },
      {
        id: "colm26-reviewer",
        date: "2026-05",
        value: 88,
        label: "COLM'26 reviewer",
        detail: "Reviewer service for COLM 2026.",
        duration: "May 2026",
        kind: "service",
        logo: "colm",
        trace: false,
        process: false
      }
    ]
  },
  {
    lane: "work",
    title: "GPU1 Work",
    subtitle: "internships and industry experiences",
    color: "#c6c6c6",
    unit: "hours",
    events: [
      {
        id: "gm-start",
        date: "2024-05",
        value: 80,
        label: "General Motors CAV Lab",
        detail: "Research intern on large-scale latency-tolerant positioning systems with Bo Yu and Fan Bai.",
        details: [
          "Designed a large-scale latency-tolerant vehicle positioning system on edge/cloud servers.",
          "Developed a deep factor graph model to handle delayed perception data while maintaining real-time responsiveness.",
          "Leveraged parallelism and prioritized scheduling to meet tight latency constraints."
        ],
        duration: "May-Aug 2024",
        kind: "job",
        logo: "gm"
      },
      { id: "gm-end", date: "2024-08", value: 0, label: "End of internship at General Motors CAV Lab", detail: "End of my internship at General Motors CAV Lab.", duration: "May-Aug 2024", kind: "end", logo: "gm" },
      {
        id: "cse589",
        date: "2024-09",
        value: 40,
        label: "Graduate Student Instructor for CSE 589",
        detail: "GSI of Advanced Computer Networks at the University of Michigan.",
        details: [
          "Led in-class discussions and held regular office hours.",
          "Delivered a guest lecture on distributed software-defined networking.",
          "Mentored graduate students on research projects, including methodology, implementation, and presentation."
        ],
        duration: "Sep-Dec 2024",
        kind: "job",
        logo: "umich"
      },
      { id: "cse589-end", date: "2024-12", value: 0, label: "End of Graduate Student Instructor for CSE 589", detail: "End of Fall 2024 CSE 589 GSI appointment.", duration: "Sep-Dec 2024", kind: "end", logo: "umich" },
      {
        id: "google-fulltime",
        date: "2025-05",
        value: 90,
        label: "Google Systems Research",
        detail: "Student Researcher in Seattle working on distributed RL frameworks for LLMs with Juncheng Gu and Arvind Krishnamurthy.",
        details: [
          "Characterized bottlenecks across the LLM RL pipeline and identified rollout as a dominant yet highly elastic component.",
          "Designed RLBoost on Google Cloud Platform to harvest fragmented spot resources, lower RL training cost, and improve overall utilization.",
          "Explored heterogeneous compute options across multi-generation GPUs and TPUs under diverse RL workloads.",
          "Contributed to an NL2SQL agentic training pipeline, optimizing multi-node communication and applying asynchronous tool calling."
        ],
        duration: "May-Aug 2025",
        kind: "job",
        logo: "google"
      },
      {
        id: "google-parttime",
        date: "2025-09",
        value: 40,
        label: "Google Systems Research part-time",
        detail: "Continued part-time work with Systems Research @ Google",
        details: [
          "Wrapped up the Systems Research @ Google student researcher internship around the RLBoost NSDI'26 acceptance.",
          "Extend RLBoost to heterogenous systems TPU+GPU.",
          "Optimize LLM inference engines and weight transfer mechanism on TPU"
        ],
        duration: "Sep 2025-Mar 2026",
        kind: "job",
        logo: "google"
      },
      { id: "google-end", date: "2026-03", value: 0, label: "End of internship at Google Systems Research", detail: "End of my part-time work during the academic term.", duration: "Sep 2025-Mar 2026", kind: "end", logo: "google" },
      {
        id: "citadel-start",
        date: "2026-06",
        value: 98,
        label: "Citadel Securities internship",
        detail: "Incoming Quantitative Researcher Intern at Citadel Securities in Miami",
        details: [
          "Incoming Quantitative Researcher Intern for summer 2026.",
          "Optimize ML infrastructures for quantitative equity analysis."
        ],
        duration: "June 2026-Aug 2026",
        kind: "job",
        logo: "citadel"
      }
    ]
  },
  {
    lane: "publications",
    title: "GPU2 Research",
    subtitle: "projects and publications",
    color: "#b4b4b4",
    unit: "engagement",
    events: [
      {
        id: "mm2gb-start",
        date: "2022-05",
        value: 28,
        ...publicationRefs("mm2gb"),
        duration: "May 2022-Nov 2024",
        kind: "start",
        logo: "mm2gb"
      },
      {
        id: "plato-start",
        date: "2024-02",
        value: 50,
        ...publicationRefs("plato"),
        duration: "Feb 2024-Aug 2025",
        kind: "start",
        logo: "plato"
      },
      {
        id: "lte-start",
        date: "2024-03",
        value: 72,
        ...publicationRefs("lte"),
        duration: "Mar 2024-Jan 2025",
        kind: "start",
        logo: "lte"
      },
      { id: "mm2gb-bcb", date: "2024-10", value: 72, label: "mm2-gb accepted to ACM BCB'24 Oral", detail: "", duration: "ACM BCB 2024", kind: "acceptance", logo: "bcb", link: "https://www.biorxiv.org/content/10.1101/2024.03.23.586366v2.abstract", trace: false },
      {
        id: "cake-start",
        date: "2024-10",
        value: 82,
        ...publicationRefs("cake"),
        duration: "Oct 2024-Jun 2025",
        kind: "start",
        logo: "cake"
      },
      { id: "mm2gb-end", date: "2024-11", value: 64, label: "mm2-gb moved to background", detail: "Main mm2-gb active project phase completed after BCB acceptance.", duration: "May 2022-Nov 2024", kind: "end", logo: "mm2gb" },
      { id: "lte-neurips", date: "2024-12", value: 64, label: "LTE accepted to NeurIPS'24 Spotlight", detail: "", duration: "NeurIPS 2024", kind: "acceptance", logo: "neurips", link: "https://arxiv.org/abs/2402.06126", trace: false },
      { id: "lte-end", date: "2025-01", value: 44, label: "LTE moved to background", detail: "Main LTE active project phase completed after NeurIPS.", duration: "Mar 2024-Jan 2025", kind: "end", logo: "lte" },
      {
        id: "hetermoe-start",
        date: "2025-01",
        value: 76,
        ...publicationRefs("hetermoe"),
        duration: "Jan 2025-present",
        kind: "start",
        logo: "hetermoe"
      },
      {
        id: "rlboost-start",
        date: "2025-05",
        value: 88,
        ...publicationRefs("rlboost"),
        duration: "May 2025-Jan 2026",
        kind: "start",
        logo: "rlboost"
      },
      { id: "cake-icml", date: "2025-05", value: 88, label: "CAKE accepted to ICML'25", detail: "", duration: "ICML 2025", kind: "acceptance", logo: "icml", link: "https://arxiv.org/abs/2410.03065", trace: false },
      { id: "cake-end", date: "2025-06", value: 70, label: "CAKE moved to background", detail: "Main CAKE active project phase completed after ICML acceptance.", duration: "Oct 2024-Jun 2025", kind: "end", logo: "cake" },
      { id: "plato-colm", date: "2025-07", value: 88, label: "Plato accepted to COLM'25", detail: "", duration: "COLM 2025", kind: "acceptance", logo: "colm", link: "https://arxiv.org/abs/2402.12280", trace: false },
      { id: "plato-end", date: "2025-08", value: 62, label: "Plato moved to background", detail: "Main Plato active project phase completed after COLM acceptance.", duration: "Feb 2024-Aug 2025", kind: "end", logo: "plato" },
      {
        id: "foundry-start",
        date: "2025-09",
        value: 96,
        ...publicationRefs("foundry"),
        duration: "Sep 2025-present",
        kind: "start",
        logo: "foundry"
      },
      { id: "rlboost-nsdi", date: "2025-12", value: 96, label: "RLBoost accepted to NSDI'26", detail: "", duration: "NSDI 2026", kind: "acceptance", logo: "nsdi", link: "https://arxiv.org/abs/2510.19225", trace: false },
      { id: "rlboost-end", date: "2026-01", value: 78, label: "RLBoost moved to background", detail: "Main RLBoost active project phase completed after NSDI acceptance.", duration: "May 2025-Jan 2026", kind: "end", logo: "rlboost" },
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
