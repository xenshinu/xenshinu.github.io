import type { BadgeKey } from "./badges";

export const profile = {
  name: "Xueshen Liu",
  nameAlt: "刘学深",
  role: "CSE Ph.D. candidate at University of Michigan",
  location: "Ann Arbor, MI",
  avatar: "/assets/images/web-logo.JPG",
  cv: "/assets/pdf/Xueshen_CV.pdf",
  scholar: "https://scholar.google.com/citations?user=3AxUf6QAAAAJ&hl=en",
  email: "mailto:liuxs@umich.edu",
  github: "https://github.com/xenshinu",
  linkedin: "https://www.linkedin.com/in/xenshinu/",
  instagram: "https://instagram.com/xenshinu.liu",
  X: "https://x.com/Xenshinu429",
  intro:
    "I build systems for cost efficient LLM training, inference, and reinforcement learning, focusing on designing elastic infrastructure to harvest heterogeneous resources.",
  focusBadges: ["llm", "infra", "eff", "elastic", "heter"] satisfies BadgeKey[],
  stack: [
    "CUDA",
    "DeepEP",
    "DeepGEMM",
    "VeRL",
    "PyTorch",
    "NCCL",
    "SGLang",
    "vLLM",
    "DeepSpeed",
    "FlashAttention",
    "LMCache",
    "Kubernetes",
    "Slurm",
    "Docker",
    "Nsight Systems",
    "Perfetto"
  ],
  links: [
    { label: "Email: liuxs AT umich.edu", href: "mailto:liuxs@umich.edu" },
    { label: "GitHub", href: "https://github.com/xenshinu" },
    { label: "LinkedIn", href: "https://www.linkedin.com/in/xenshinu/"},
    { label: "X (Twitter)", href: "https://x.com/Xenshinu429" },
    { label: "Instagram", href: "https://instagram.com/xenshinu.liu" },
    { label: "CV", href: "/assets/pdf/Xueshen_CV.pdf" }
  ]
};
