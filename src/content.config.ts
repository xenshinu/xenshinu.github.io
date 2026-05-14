import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    date: z.union([z.string(), z.date()]),
    author_profile: z.boolean().optional(),
    comments: z.boolean().optional(),
    tags: z.array(z.string()).default([]),
    summary: z.string().optional()
  })
});

export const collections = { posts };
