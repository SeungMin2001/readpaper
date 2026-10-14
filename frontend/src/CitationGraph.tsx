import type { Source } from './types'

type Props = { title: string; sources: Source[] }
type GraphNode = { source: Source; x: number; y: number; side: 'reference' | 'citing' }

function shortTitle(value: string, max = 29) {
  return value.length > max ? `${value.slice(0, max - 1).trimEnd()}…` : value
}

export default function CitationGraph({ title, sources }: Props) {
  const references = sources.filter((source) => source.kind === 'referenced')
  const citing = sources.filter((source) => source.kind === 'citing')
  const shownReferences = references.slice(0, 8)
  const shownCiting = citing.slice(0, 6)
  const rowCount = Math.max(shownReferences.length, shownCiting.length, 1)
  const height = Math.max(350, 82 + rowCount * 48)
  const centerY = height / 2
  const rowY = (index: number, count: number) => count < 2 ? centerY : 40 + index * ((height - 80) / (count - 1))

  const nodes: GraphNode[] = [
    ...shownReferences.map((source, index) => ({ source, x: 54, y: rowY(index, shownReferences.length), side: 'reference' as const })),
    ...shownCiting.map((source, index) => ({ source, x: 746, y: rowY(index, shownCiting.length), side: 'citing' as const })),
  ]

  return <div className="citation-graph-scroll">
    {references.length + citing.length === 0 ? <div className="empty-graph"><span>—</span><p>확인된 인용 관계가 아직 없습니다.</p><small>외부 검색에서 논문을 찾지 못했거나, 참고문헌 정보가 제공되지 않았을 수 있습니다.</small></div> : <>
      <svg className="citation-graph" viewBox={`0 0 800 ${height}`} role="img" aria-label={`${title}의 직접 인용 관계 그래프`}>
        <defs>
          <marker id="arrow-reference" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3.5 L0,7 z" fill="#8a9a88" /></marker>
          <marker id="arrow-citing" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3.5 L0,7 z" fill="#99a9b4" /></marker>
        </defs>
        <text className="graph-column-title" x="54" y="22">REFERENCES</text>
        <text className="graph-column-title graph-column-title--right" x="746" y="22" textAnchor="end">CITES THIS PAPER</text>
        {nodes.map((node) => {
          const rootEdgeX = node.side === 'reference' ? 308 : 492
          const nodeEdgeX = node.side === 'reference' ? 61 : 739
          const curve = node.side === 'reference'
            ? `M ${rootEdgeX} ${centerY} C ${rootEdgeX - 90} ${centerY}, ${nodeEdgeX + 110} ${node.y}, ${nodeEdgeX} ${node.y}`
            : `M ${nodeEdgeX} ${node.y} C ${nodeEdgeX - 110} ${node.y}, ${rootEdgeX + 90} ${centerY}, ${rootEdgeX} ${centerY}`
          return <path key={`edge-${node.source.id}`} className={`graph-edge graph-edge--${node.side}`} d={curve} markerEnd={`url(#arrow-${node.side})`} />
        })}
        {nodes.map((node) => <g className={`graph-node graph-node--${node.side}`} key={node.source.id}>
          <title>{node.source.title}{node.source.year ? ` (${node.source.year})` : ''}</title>
          <circle cx={node.x} cy={node.y} r="7" />
          <text x={node.side === 'reference' ? 72 : 728} y={node.y - 3} textAnchor={node.side === 'reference' ? 'start' : 'end'}>{shortTitle(node.source.title)}</text>
          <text className="graph-node-year" x={node.side === 'reference' ? 72 : 728} y={node.y + 12} textAnchor={node.side === 'reference' ? 'start' : 'end'}>{node.source.year || '연도 미상'}</text>
        </g>)}
        <g className="graph-root">
          <rect x="310" y={centerY - 43} width="180" height="86" rx="13" />
          <text className="graph-root-label" x="400" y={centerY - 13} textAnchor="middle">CURRENT PAPER</text>
          <text className="graph-root-title" x="400" y={centerY + 9} textAnchor="middle">{shortTitle(title, 26)}</text>
          <text className="graph-root-count" x="400" y={centerY + 27} textAnchor="middle">{references.length} found · {citing.length} citing</text>
        </g>
      </svg>
      <div className="graph-caption"><span><i className="graph-caption-reference" />현재 논문이 인용한 연구</span><span><i className="graph-caption-citing" />현재 논문을 인용한 연구</span></div>
      {(references.length > shownReferences.length || citing.length > shownCiting.length) && <p className="graph-limit-note">그래프에는 각 방향의 일부 관계를 표시합니다. 전체 목록은 아래에서 확인할 수 있어요.</p>}
    </>}
  </div>
}
