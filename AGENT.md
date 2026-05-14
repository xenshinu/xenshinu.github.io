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
- Process rows should show date, duration, and event only. Do not show raw event type or trace value there.
- Marker clicks should open an in-panel bubble/popover, not navigate away or open a modal.
- Each trace panel should auto-fit its own x-axis date range and y-axis range.
- Do not split a GPU into multiple project lines. Merge the story into one trace and use markers for role switches, starts, endings, acceptances, and posts.
- Mobile must use tap/click reveal behavior. Do not make content hover-only.
- Local badge-style logos are preferred over externally fetched logo images.

## Content Model

Dashboard timeline data lives in `src/data/timeline.ts`. Edit `laneTraces` for the live dashboard.

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
    duration?: string;
    kind: "start" | "end" | "release" | "acceptance" | "job" | "post" | "award" | "talk" | "service" | "education" | "paper" | "blog";
    logo?: string;
    link?: string;
    trace?: boolean;     // false means marker-only; it does not affect the line
  }>;
};
```

Blog posts live in `src/content/posts`. The route `/posts/[slug]/` is the canonical post route. Root-level legacy post slugs redirect to canonical post URLs.

## Routes

- `/`: main dashboard.
- `/blog/`: blog process archive.
- `/research/`: publications and projects.
- `/tags/`: tag archive.
- `/links/`: external links and profile links.
- `/About/`, `/about/`, `/Blogs/`, and `/Publications/`: compatibility redirects.

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

- [ ] Timeline panels always show trace up to today's month
- [ ] only show latest 3 events in the process scroll window, when hover on the trace, it should automatically scroll to the corresponding event and highlight.
- [ ] when hover a event in the process scroll window, highlight both the corresponding segments and turning points in the trace
- [ ] use terminal style colors instead of pure white/black, but keep it simple, use minimum number of colors
- [ ] in preview of panels, all event should be dots instead of rounded bars.
- [ ] No need to show both start and end and acceptance on the process window, the process entries only include the event itself with id, duration, name, (title) and one sentence brief introduction. And if reader click it, unfold a dropdown box listing the details like mentor and information in bullet points, when click other event, fold the previous one.
- [ ] All acceptance and end event doesn't need a details in the pop window, just keep its details empty.
- [ ] Enable horizontal scrolling of timeline panel when it is too wide or too much events. And by default it should shows the most up to date part
- [ ] Optimize for vertical screen or phone viewers.

