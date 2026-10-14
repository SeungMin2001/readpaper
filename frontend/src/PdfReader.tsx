import { useEffect, useMemo, useRef, useState } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import type { PaperPage, Word, WordGloss } from './types'

type GlossReference = { key: string; gloss: WordGloss; first: boolean }

type Props = {
  document: PDFDocumentProxy
  pageIndex: number
  page?: PaperPage
  wordGlosses: Record<string, WordGloss>
  translated: Set<string>
  pendingWordIds: Set<string>
  onWord: (word: Word, gloss?: WordGloss) => void
}

export default function PdfReader({
  document, pageIndex, page, wordGlosses, translated, pendingWordIds, onWord,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState(1)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [renderError, setRenderError] = useState('')

  useEffect(() => {
    const frame = frameRef.current
    const canvas = canvasRef.current
    if (!frame || !canvas) return
    let cancelled = false
    let renderTask: ReturnType<Awaited<ReturnType<PDFDocumentProxy['getPage']>>['render']> | null = null
    let rendering = false
    let requested = false

    async function renderPage() {
      if (!frame || !canvas) return
      if (rendering) return
      rendering = true
      while (requested && !cancelled) {
        requested = false
        try {
          const pdfPage = await document.getPage(pageIndex + 1)
          if (cancelled) break
          const width = Math.min(frame.clientWidth, 920)
          const baseViewport = pdfPage.getViewport({ scale: 1 })
          setDimensions({ width: baseViewport.width, height: baseViewport.height })
          const nextScale = width / (page?.width || baseViewport.width)
          const viewport = pdfPage.getViewport({ scale: nextScale })
          const pixelRatio = Math.min(window.devicePixelRatio || 1, 2)
          canvas.width = Math.floor(viewport.width * pixelRatio)
          canvas.height = Math.floor(viewport.height * pixelRatio)
          canvas.style.width = `${viewport.width}px`
          canvas.style.height = `${viewport.height}px`
          const context = canvas.getContext('2d')
          if (!context) throw new Error('PDF 화면을 그릴 수 없습니다.')
          renderTask = pdfPage.render({
            canvas, canvasContext: context, viewport,
            transform: [pixelRatio, 0, 0, pixelRatio, 0, 0],
          })
          await renderTask.promise
          renderTask = null
          if (!cancelled) {
            setScale(nextScale)
            setRenderError('')
          }
        } catch (error) {
          if (!cancelled && !(error instanceof Error && error.name === 'RenderingCancelledException')) {
            setRenderError('PDF 페이지를 표시하지 못했습니다.')
          }
        }
      }
      rendering = false
    }

    const observer = new ResizeObserver(() => {
      requested = true
      renderTask?.cancel()
      void renderPage()
    })
    observer.observe(frame)
    return () => {
      cancelled = true
      observer.disconnect()
      renderTask?.cancel()
    }
  }, [document, pageIndex, page?.index, page?.width])

  const byWord = useMemo(() => {
    const map = new Map<string, GlossReference>()
    for (const [wordId, gloss] of Object.entries(wordGlosses)) {
      const key = gloss.word_ids.join(':')
      gloss.word_ids.forEach((id, wordIndex) => map.set(id, { key, gloss, first: wordIndex === 0 }))
      if (!gloss.word_ids.includes(wordId)) map.set(wordId, { key, gloss, first: false })
    }
    return map
  }, [wordGlosses])

  return (
    <div ref={frameRef} className="pdf-frame">
      <div className="pdf-page" style={{ width: (page?.width || dimensions.width) * scale, height: (page?.height || dimensions.height) * scale }}>
        <canvas ref={canvasRef} className="pdf-canvas" aria-label={`논문 ${pageIndex + 1}페이지`} />
        {(page?.words || []).map((word) => {
          const ref = byWord.get(word.id)
          const active = Boolean(ref && translated.has(ref.key))
          const pending = pendingWordIds.has(word.id)
          const width = Math.max((word.x1 - word.x0) * scale, 8)
          const height = Math.max((word.y1 - word.y0) * scale, 8)
          return (
            <button
              key={word.id}
              type="button"
              className={[
                'pdf-word', active ? 'pdf-word--translated' : '', pending ? 'pdf-word--pending' : '',
                active && ref?.first ? 'pdf-word--first' : '',
              ].join(' ')}
              style={{
                left: word.x0 * scale,
                top: word.y0 * scale,
                width: active && ref?.first ? 'max-content' : width,
                minWidth: width,
                height,
                fontSize: Math.max(8, Math.min(height * 0.86, 15)),
              }}
              title={pending ? '문맥에서 용어를 찾고 있어요' : ref ? `${ref.gloss.original} 뜻 보기` : `${word.text} 문맥 뜻 보기`}
              aria-label={`${word.text}의 문맥 뜻 보기`}
              disabled={pending}
              onClick={() => onWord(word, ref?.gloss)}
            >
              {active && ref?.first ? ref.gloss.meaning : ''}
            </button>
          )
        })}
      </div>
      {renderError && <p className="inline-error">{renderError}</p>}
    </div>
  )
}
