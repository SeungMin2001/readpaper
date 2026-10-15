import type { Source } from "./types";

type Props = { title: string; sources: Source[] };
export default function CitationGraph({ title, sources }: Props) {
  const references = sources.filter((s) => s.kind === "referenced");
  const others = sources.filter((s) =>
    ["citing", "related", "recent"].includes(s.kind),
  );
  const height = Math.max(
    400,
    Math.max(references.length, others.length) * 64 + 100,
  );
  const nodes = [
    {
      id: "root",
      title,
      x: 470,
      y: height / 2,
      kind: "paper",
      url: null as string | null,
      openalex_id: "",
    },
    ...references.map((s, i) => ({ ...s, x: 150, y: 65 + i * 64 })),
    ...others.map((s, i) => ({ ...s, x: 790, y: 65 + i * 64 })),
  ];
  const edges = new Set<string>();
  references.forEach((s) => edges.add(`root|${s.id}`));
  others
    .filter((s) => s.kind === "citing")
    .forEach((s) => edges.add(`${s.id}|root`));
  const byWork = new Map(
    nodes.filter((n) => n.openalex_id).map((n) => [n.openalex_id, n.id]),
  );
  sources.forEach((source) =>
    source.referenced_works?.forEach((work) => {
      const target = byWork.get(work);
      if (target && target !== source.id) edges.add(`${source.id}|${target}`);
    }),
  );
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const short = (text: string) =>
    text.length > 34 ? text.slice(0, 33) + "…" : text;
  if (nodes.length === 1)
    return (
      <div className="empty-graph">
        <p>확인된 논문 관계가 없습니다.</p>
      </div>
    );
  return (
    <div className="citation-graph-scroll">
      <svg
        className="citation-graph"
        viewBox={`0 0 940 ${height}`}
        role="img"
        aria-label="논문 인용 관계 그래프"
      >
        <defs>
          <marker
            id="citation-arrow"
            markerWidth="8"
            markerHeight="8"
            refX="8"
            refY="4"
            orient="auto"
          >
            <path d="M0 0 L8 4 L0 8" fill="#a1a1aa" />
          </marker>
        </defs>
        <text x="35" y="26" className="graph-column-title">
          참고문헌 {references.length}
        </text>
        <text x="670" y="26" className="graph-column-title">
          피인용 · 관련 연구 {others.length}
        </text>
        {[...edges].map((edge) => {
          const [a, b] = edge.split("|"),
            from = byId.get(a),
            to = byId.get(b);
          if (!from || !to) return null;
          const direction = from.x < to.x ? 1 : -1;
          return (
            <path
              key={edge}
              d={`M ${from.x + direction * 116} ${from.y} C ${(from.x + to.x) / 2} ${from.y}, ${(from.x + to.x) / 2} ${to.y}, ${to.x - direction * 118} ${to.y}`}
              fill="none"
              stroke="#c5c5c5"
              strokeWidth="1.2"
              markerEnd="url(#citation-arrow)"
            />
          );
        })}
        {nodes.map((node) => (
          <a
            key={node.id}
            href={node.url || undefined}
            target="_blank"
            rel="noopener noreferrer"
          >
            <g
              className={`graph-node ${node.id === "root" ? "graph-root" : ""}`}
            >
              <title>{node.title}</title>
              <rect
                x={node.x - 115}
                y={node.y - 22}
                width="230"
                height="44"
                rx="6"
                fill={node.id === "root" ? "#37352f" : "#fff"}
                stroke="#deddd9"
              />
              <circle
                cx={node.x - 100}
                cy={node.y}
                r="4"
                fill={
                  node.kind === "related"
                    ? "#c9a56b"
                    : node.kind === "citing"
                      ? "#7ba0b7"
                      : "#8d9d8d"
                }
              />
              <text
                x={node.x - 88}
                y={node.y + 4}
                style={{
                  fontSize: 11,
                  fill: node.id === "root" ? "#fff" : "#37352f",
                }}
              >
                {short(node.title)}
              </text>
            </g>
          </a>
        ))}
      </svg>
      <p className="graph-caption">
        화살표: 인용하는 논문 → 인용된 논문 · 인용이 확인되지 않은 관련 연구는
        선 없이 표시합니다.
      </p>
    </div>
  );
}
