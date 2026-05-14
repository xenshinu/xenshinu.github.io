import { formatDate } from "./date";
import { getCollection, type CollectionEntry } from "astro:content";

function slugFromId(id: string): string {
  return id
    .split("/")
    .pop()!
    .replace(/\.md$/, "")
    .replace(/^\d{4}-\d{2}-\d{2}-/, "");
}

function normalizeTags(tags: unknown): string[] {
  if (!Array.isArray(tags)) return [];
  return tags.map((tag) => String(tag));
}

function rawSummary(raw: string | undefined): string {
  if (!raw) return "";
  const body = raw.replace(/^---[\s\S]*?---/, "").trim();
  const paragraph = body
    .split(/\n{2,}/)
    .map((chunk) => chunk.replace(/[#>*_`\[\]()]/g, "").trim())
    .find((chunk) => chunk.length > 60);

  return paragraph ? `${paragraph.slice(0, 180)}${paragraph.length > 180 ? "..." : ""}` : "";
}

export type PostEntry = {
  entry: CollectionEntry<"posts">;
  slug: string;
  title: string;
  date: Date;
  dateLabel: string;
  tags: string[];
  summary: string;
};

export async function getAllPosts(): Promise<PostEntry[]> {
  const entries = await getCollection("posts");

  return entries
    .map((entry) => {
      const dateValue = entry.data.date;
      const date = dateValue instanceof Date ? dateValue : new Date(String(dateValue));
      const slug = slugFromId(entry.id);

      return {
        entry,
        slug,
        title: String(entry.data.title ?? slug),
        date,
        dateLabel: formatDate(date),
        tags: normalizeTags(entry.data.tags),
        summary: String(entry.data.summary ?? rawSummary(entry.body))
      };
    })
    .sort((a, b) => b.date.getTime() - a.date.getTime());
}

export async function getTags(): Promise<Array<{ tag: string; posts: PostEntry[] }>> {
  const posts = await getAllPosts();
  const tags = new Map<string, PostEntry[]>();

  for (const post of posts) {
    for (const tag of post.tags) {
      tags.set(tag, [...(tags.get(tag) ?? []), post]);
    }
  }

  return Array.from(tags.entries())
    .map(([tag, taggedPosts]) => ({ tag, posts: taggedPosts }))
    .sort((a, b) => a.tag.localeCompare(b.tag));
}
