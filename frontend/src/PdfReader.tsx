import { useEffect, useMemo, useRef, useState } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import type { PaperPage, Sentence, SentenceTranslation, TranslationUnit, Word } from './types'

type UnitReference = { key: string; unit: TranslationUnit; first: boolean }

type Props = {
  document: PDFDocumentProxy
  page: PaperPage
  sentences: Sentence[]
  translations: Record<string, SentenceTranslation>
  translated: Set<string>
  selectedSentence: string | null
  mode: 'word' | 'sentence'
  onWord: (key: string, unit: TranslationUnit, word: Word) => void
  onSentence: (id: string) => void
}

export default function PdfReader({
  document, page, sentences, translations, translated,
  selectedSentence, mode, onWord, onSentence,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState(1)
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
          const pdfPage = await document.getPage(page.index + 1)
          if (cancelled) break
          const width = Math.min(frame.clientWidth, 920)
          const nextScale = width / page.width
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
  }, [document, page.index, page.width])

  const byWord = useMemo(() => {
    const map = new Map<string, UnitReference>()
    for (const sentence of sentences) {
      const translation = translations[sentence.id]
      if (!translation) continue
      translation.units.forEach((unit, index) => {
        const key = `${sentence.id}:${index}`
        unit.word_ids.forEach((id, wordIndex) => {
          map.set(id, { key, unit, first: wordIndex === 0 })
        })
      })
    }
    return map
  }, [sentences, translations])

  return (
    <div ref={frameRef} className="pdf-frame">
      <div className="pdf-page" style={{ width: page.width * scale, height: page.height * scale }}>
        <canvas ref={canvasRef} className="pdf-canvas" aria-label={`논문 ${page.index + 1}페이지`} />
        {page.words.map((word) => {
          const ref = byWord.get(word.id)
          const active = Boolean(ref && translated.has(ref.key))
          const isSelected = selectedSentence === word.sentence_id
          const width = Math.max((word.x1 - word.x0) * scale, 8)
          const height = Math.max((word.y1 - word.y0) * scale, 8)
          return (
            <button
              key={word.id}
              type="button"
              className={[
                'pdf-word', active ? 'pdf-word--translated' : '',
                active && ref?.first ? 'pdf-word--first' : '',
                isSelected ? 'pdf-word--selected' : '',
              ].join(' ')}
              style={{
                left: word.x0 * scale,
                top: word.y0 * scale,
                width: active && ref?.first ? 'max-content' : width,
                minWidth: width,
                height,
                fontSize: Math.max(8, Math.min(height * 0.86, 15)),
              }}
              title={mode === 'sentence' ? '이 문장 번역 보기' : ref ? `${ref.unit.original} 뜻 보기` : word.text}
              aria-label={mode === 'sentence' ? `${word.text}가 포함된 문장 선택` : `${word.text} 뜻 보기`}
              onClick={() => {
                if (mode === 'sentence') onSentence(word.sentence_id)
                else if (ref) onWord(ref.key, ref.unit, word)
              }}
            >
              {active && ref?.first ? ref.unit.meaning : ''}
            </button>
          )
        })}
      </div>
      {renderError && <p className="inline-error">{renderError}</p>}
    </div>
  )
}
