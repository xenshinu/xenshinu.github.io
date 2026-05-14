# XUESHEN-SMI Site Notes

This repo is an Astro + TypeScript rebuild of Xueshen Liu's personal site. The visual model is a simplified terminal-style `nvtop` / `nvidia-smi` inspired research dashboard.

## Design Requirements

- The home page is the primary interactive dashboard.
- Keep the visual language close to GPU observability: trace panels, utilization lines, process/news rows, compact metadata, and technical badges.
- Do not hardcode trace SVG art. Timeline traces must be editable through TypeScript data files.
- Each GPU panel renders exactly one trace from `laneTraces`. Turning points are events with editable `value` fields.
- Events can set `trace: false` to become marker-only rows, useful for awards, service, paper acceptances, and releases that should not change y-value.
- Trace markers should show only compact logo/text tokens on the chart. Full event title, date, detail, and links belong in the click popover and process row.
- `kind: "end"` renders as a small unlabeled circle. `kind: "acceptance"` renders near the x-axis to avoid colliding with project tokens.
- The blog GPU trace is cumulative post count.
- Hover/click sync matters: markers, traces, selected GPU panels, and process rows should highlight each other.
- Do not use modal zoom windows for the trace grid. Clicking a GPU panel should expand it in place and compress the other GPU panels underneath.
- The process window sits above the GPU panels and should show only rows for the currently selected GPU/lane.
- Process rows should show numeric id, time, and event only. Do not show raw event type or trace value there.
- Marker clicks should open an in-panel bubble/popover, not navigate away or open a modal.
- The home page should not add a second research-thread section below the dashboard. Use a compact `Host Processes` news list for general updates instead.
- Each trace panel should auto-fit its own x-axis date range and y-axis range.
- Do not split a GPU into multiple project lines. Merge the story into one trace and use markers for role switches, starts, endings, acceptances, and posts.
- Mobile must use tap/click reveal behavior. Do not make content hover-only.
- Local badge-style logos are preferred over externally fetched logo images.

## Content Model

Dashboard timeline data lives in `src/data/timeline.ts`. Edit `laneTraces` for the live dashboard.
Publication start-event copy in the timeline should reuse `src/data/publications.ts` through `publicationRefs(...)`, so popover summaries and process dropdown bullets stay synced with the research page.

```ts
type LaneTrace = {
  lane: "education" | "work" | "publications" | "blogs";
  title: string;
  subtitle?: string;
  color: string;
  unit: string;
  events: Array<{
    id: string;
    date: string;
    value: number;
    label: string;
    detail: string;
    details?: string[];  // optional bullet list for process dropdowns
    duration?: string;
    kind: "start" | "end" | "release" | "acceptance" | "job" | "post" | "award" | "talk" | "service" | "education" | "paper" | "blog";
    logo?: string;
    link?: string;
    trace?: boolean;     // false means marker-only; it does not affect the line
    process?: boolean;   // false hides timeline-only markers from the process window
  }>;
};
```

Blog posts live in `src/content/posts`. The route `/posts/[slug]/` is the canonical post route.

## Routes

- `/`: main dashboard.
- `/blog/`: blog process archive.
- `/research/`: publications and projects.
- `/tags/`: tag archive.
- `/links/`: external links and profile links.

## Deployment

Target repository: `xenshinu/xenshinu.github.io`.

The site is static and deploys with GitHub Actions Pages from `.github/workflows/deploy.yml`. Use:

```bash
npm install
npm run build
npm run dev
```

GitHub Pages must be configured to use GitHub Actions as the publishing source.


## Next Steps from user

- [x] When user click the preview trace panel, the page should scroll to the position where the process window is on the top
- [x] In phone mode, the process window, the host processses and memory heap width doesn't match the other window, usually a bit wider
- [x] In phone view, only the education proceses details is normal small size, the other (e.g. work) the drop down information get very big
- [x] In phone view, the TIME and WORKLOAD column of processes window will overlap, there should be more space for them while the ID column takes too much space, the column names should also move with the horizontal scroll bar, otherwise the content is mismatch.
- [x] Change the order of columns in processes window to ID, WORKLOAD, TIME
- [x] Also skip the second left most time of the trace x-axis label if the first label is too close with the second
- [x] For host processes, put type logo under the time instead of creating a new column
