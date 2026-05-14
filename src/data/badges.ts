export type BadgeKey =
  | "umich"
  | "sjtu"
  | "google"
  | "citadel"
  | "gm"
  | "amd"
  | "cuda"
  | "llm"
  | "infra"
  | "heter"
  | "eff"
  | "elastic"
  | "triton"
  | "pytorch"
  | "vllm"
  | "nccl"
  | "docker"
  | "claude"
  | "nsdi"
  | "iclr"
  | "icml"
  | "colm"
  | "neurips"
  | "bcb"
  | "arxiv"
  | "foundry"
  | "rlboost"
  | "hetermoe"
  | "plato"
  | "cake"
  | "lte"
  | "mm2gb";

export type Badge = {
  key: BadgeKey;
  label: string;
  short: string;
  tone: string;
};

export const badges: Record<BadgeKey, Badge> = {
  umich: { key: "umich", label: "University of Michigan", short: "UMich", tone: "maize" },
  sjtu: { key: "sjtu", label: "Shanghai Jiao Tong University", short: "SJTU", tone: "red" },
  google: { key: "google", label: "Google", short: "Google", tone: "blue" },
  citadel: { key: "citadel", label: "Citadel Securities", short: "CitSec", tone: "gold" },
  gm: { key: "gm", label: "General Motors", short: "GM", tone: "cyan" },
  amd: { key: "amd", label: "AMD", short: "AMD", tone: "red" },
  cuda: { key: "cuda", label: "CUDA", short: "CUDA", tone: "green" },
  llm: { key: "llm", label: "Large Language Model", short: "LLM", tone: "orange" },
  infra: { key: "infra", label: "Infrastructure", short: "Infra", tone: "gold" },
  elastic: { key: "elastic", label: "Elasticity", short: "Elastic", tone: "blue" },
  eff: { key: "eff", label: "Efficiency", short: "Efficient", tone: "green" },
  heter: { key: "heter", label: "Heterogenity", short: "Heter", tone: "purple" },
  triton: { key: "triton", label: "Triton", short: "Triton", tone: "purple" },
  pytorch: { key: "pytorch", label: "PyTorch", short: "PyTorch", tone: "orange" },
  vllm: { key: "vllm", label: "vLLM", short: "vLLM", tone: "blue" },
  nccl: { key: "nccl", label: "NCCL", short: "NCCL", tone: "green" },
  docker: { key: "docker", label: "Docker", short: "Docker", tone: "blue" },
  claude: { key: "claude", label: "Claude Code", short: "Claude", tone: "orange" },
  nsdi: { key: "nsdi", label: "NSDI", short: "NSDI", tone: "green" },
  iclr: { key: "iclr", label: "ICLR", short: "ICLR", tone: "purple" },
  icml: { key: "icml", label: "ICML", short: "ICML", tone: "purple" },
  colm: { key: "colm", label: "COLM", short: "COLM", tone: "cyan" },
  neurips: { key: "neurips", label: "NeurIPS", short: "NIPS", tone: "purple" },
  bcb: { key: "bcb", label: "ACM BCB", short: "BCB", tone: "red" },
  arxiv: { key: "arxiv", label: "arXiv", short: "arXiv", tone: "red" },
  foundry: { key: "foundry", label: "Foundry", short: "Foundry", tone: "cyan" },
  rlboost: { key: "rlboost", label: "RLBoost", short: "RLBoost", tone: "green" },
  hetermoe: { key: "hetermoe", label: "HeterMoE", short: "HeterMoE", tone: "purple" },
  plato: { key: "plato", label: "Plato", short: "Plato", tone: "blue" },
  cake: { key: "cake", label: "CAKE", short: "CAKE", tone: "gold" },
  lte: { key: "lte", label: "LTE", short: "LTE", tone: "orange" },
  mm2gb: { key: "mm2gb", label: "mm2-gb", short: "MM2-gb", tone: "green" }
};
