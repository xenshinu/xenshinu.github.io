import type { BadgeKey } from "./badges";

export const profile = {
  name: "Xueshen Liu",
  nameAlt: "刘学深",
  role: "CSE Ph.D. candidate at University of Michigan",
  location: "Ann Arbor, MI",
  avatar: "/assets/images/web-logo.JPG",
  cv: "/assets/pdf/Xueshen_CV.pdf",
  email: "mailto:liuxs@umich.edu",
  github: "https://github.com/xenshinu",
  instagram: "https://instagram.com/xenshinu.liu",
  intro:
    "I build systems for efficient LLM training, inference, and reinforcement learning, with a focus on GPU parallel computing, CUDA graph runtimes, KV-cache systems, and elastic heterogeneous infrastructure.",
  focusBadges: ["llm", "infra", "eff", "elastic", "heter"] satisfies BadgeKey[],
  stack: [
    "VeRL",
    "PyTorch",
    "DeepSpeed",
    "NCCL",
    "SGLang",
    "vLLM",
    "FlashAttention",
    "LMCache",
    "CUTLASS",
    "Kubernetes",
    "Slurm",
    "Docker",
    "Nsight Systems",
    "Perfetto"
  ],
  links: [
    { label: "Email", href: "mailto:liuxs@umich.edu" },
    { label: "GitHub", href: "https://github.com/xenshinu" },
    { label: "Instagram", href: "https://instagram.com/xenshinu.liu" },
    { label: "CV", href: "/assets/pdf/Xueshen_CV.pdf" }
  ]
};
