export function parseMonth(value: string | Date): number {
  if (value instanceof Date) {
    return value.getUTCFullYear() * 12 + value.getUTCMonth();
  }

  const match = String(value).match(/^(\d{4})(?:-(\d{2}))?/);
  if (!match) return 0;
  return Number(match[1]) * 12 + Number(match[2] ?? "1") - 1;
}

export function formatMonth(value: string | Date, style: "short" | "long" = "short"): string {
  const date = value instanceof Date ? value : new Date(`${value.length === 7 ? `${value}-01` : value}T00:00:00Z`);
  return new Intl.DateTimeFormat("en", {
    month: style,
    year: "numeric",
    timeZone: "UTC"
  }).format(date);
}

export function formatDate(value: string | Date): string {
  const date = value instanceof Date ? value : new Date(`${value}T00:00:00Z`);
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC"
  }).format(date);
}
