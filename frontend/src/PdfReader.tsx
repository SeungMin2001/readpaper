import { useEffect, useMemo, useRef, useState } from "react";
import type { PDFDocumentProxy } from "pdfjs-dist";
import type { Attachment, PaperPage, Word, WordGloss } from "./types";

type GlossReference = { key: string; gloss: WordGloss; first: boolean };

type Props = {
  document: PDFDocumentProxy;
  pageIndex: number;
  page?: PaperPage;
  wordGlosses: Record<string, WordGloss>;
  translated: Set<string>;
  pendingWordIds: Set<string>;
  onWord: (word: Word, gloss?: WordGloss) => void;
  onAttach: (attachment: Attachment) => void;
};

export default function PdfReader({
  document,
  pageIndex,
  page,
  wordGlosses,
  translated,
  pendingWordIds,
  onWord,
  onAttach,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [mode, setMode] = useState<"read" | "text" | "figure">("read");
  const [selection, setSelection] = useState<Attachment | null>(null);
  const [box, setBox] = useState<number[] | null>(null);
  const start = useRef<{ x: number; y: number; word?: string } | null>(null);
  useEffect(() => {
    setSelection(null);
    setBox(null);
  }, [pageIndex]);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [renderError, setRenderError] = useState("");

  useEffect(() => {
    const frame = frameRef.current;
    const canvas = canvasRef.current;
    if (!frame || !canvas) return;
    let cancelled = false;
    let renderTask: ReturnType<
      Awaited<ReturnType<PDFDocumentProxy["getPage"]>>["render"]
    > | null = null;
    let rendering = false;
    let requested = false;

    async function renderPage() {
      if (!frame || !canvas) return;
      if (rendering) return;
      rendering = true;
      while (requested && !cancelled) {
        requested = false;
        try {
          const pdfPage = await document.getPage(pageIndex + 1);
          if (cancelled) break;
          const width = Math.min(frame.clientWidth, 920);
          const baseViewport = pdfPage.getViewport({ scale: 1 });
          setDimensions({
            width: baseViewport.width,
            height: baseViewport.height,
          });
          const nextScale = width / (page?.width || baseViewport.width);
          const viewport = pdfPage.getViewport({ scale: nextScale });
          const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
          canvas.width = Math.floor(viewport.width * pixelRatio);
          canvas.height = Math.floor(viewport.height * pixelRatio);
          canvas.style.width = `${viewport.width}px`;
          canvas.style.height = `${viewport.height}px`;
          const context = canvas.getContext("2d");
          if (!context) throw new Error("PDF 화면을 그릴 수 없습니다.");
          renderTask = pdfPage.render({
            canvas,
            canvasContext: context,
            viewport,
            transform: [pixelRatio, 0, 0, pixelRatio, 0, 0],
          });
          await renderTask.promise;
          renderTask = null;
          if (!cancelled) {
            setScale(nextScale);
            setRenderError("");
          }
        } catch (error) {
          if (
            !cancelled &&
            !(
              error instanceof Error &&
              error.name === "RenderingCancelledException"
            )
          ) {
            setRenderError("PDF 페이지를 표시하지 못했습니다.");
          }
        }
      }
      rendering = false;
    }

    const observer = new ResizeObserver(() => {
      requested = true;
      renderTask?.cancel();
      void renderPage();
    });
    observer.observe(frame);
    return () => {
      cancelled = true;
      observer.disconnect();
      renderTask?.cancel();
    };
  }, [document, pageIndex, page?.index, page?.width]);

  const byWord = useMemo(() => {
    const map = new Map<string, GlossReference>();
    for (const [wordId, gloss] of Object.entries(wordGlosses)) {
      const key = gloss.word_ids.join(":");
      gloss.word_ids.forEach((id, wordIndex) =>
        map.set(id, { key, gloss, first: wordIndex === 0 }),
      );
      if (!gloss.word_ids.includes(wordId))
        map.set(wordId, { key, gloss, first: false });
    }
    return map;
  }, [wordGlosses]);

  const words = [...(page?.words || [])].sort(
    (a, b) =>
      Number(a.sentence_id.slice(1)) - Number(b.sentence_id.slice(1)) ||
      Number(a.id.slice(1)) - Number(b.id.slice(1)),
  );
  const overlays = [
    ...new Map([...byWord.values()].map((ref) => [ref.key, ref])).values(),
  ]
    .filter((ref) => translated.has(ref.key))
    .flatMap((ref) => {
      const members = words.filter((word) =>
        ref.gloss.word_ids.includes(word.id),
      );
      const lines: Word[][] = [];
      for (const word of members) {
        const line = lines.find((items) => Math.abs(items[0].y0 - word.y0) < 4);
        if (line) line.push(word);
        else lines.push([word]);
      }
      const bounds = lines.map((line) => ({
        x: Math.min(...line.map((w) => w.x0)),
        y: Math.min(...line.map((w) => w.y0)),
        right: Math.max(...line.map((w) => w.x1)),
        bottom: Math.max(...line.map((w) => w.y1)),
        line,
      }));
      const widest = bounds.reduce(
        (best, b, i) =>
          b.right - b.x > (bounds[best]?.right - bounds[best]?.x || 0)
            ? i
            : best,
        0,
      );
      return bounds.map((b, i) => ({
        ...b,
        ref,
        label: i === widest ? ref.gloss.meaning : "",
      }));
    });
  function point(event: React.PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min((event.clientX - rect.left) / scale, page?.width || 0),
      ),
      y: Math.max(
        0,
        Math.min((event.clientY - rect.top) / scale, page?.height || 0),
      ),
    };
  }
  function makeSelection(event: React.PointerEvent<HTMLDivElement>) {
    if (!start.current || !page) return;
    const end = point(event),
      origin = start.current;
    start.current = null;
    const rect = [
      Math.min(origin.x, end.x),
      Math.min(origin.y, end.y),
      Math.max(origin.x, end.x),
      Math.max(origin.y, end.y),
    ];
    const selected = words.filter(
      (w) =>
        (w.x0 + w.x1) / 2 >= rect[0] &&
        (w.x0 + w.x1) / 2 <= rect[2] &&
        (w.y0 + w.y1) / 2 >= rect[1] &&
        (w.y0 + w.y1) / 2 <= rect[3],
    );
    if (mode === "figure") {
      if (
        rect[2] - rect[0] < 5 ||
        rect[3] - rect[1] < 5 ||
        !canvasRef.current
      ) {
        setBox(null);
        return;
      }
      const canvas = canvasRef.current,
        crop = window.document.createElement("canvas");
      const ratio = canvas.width / page.width;
      crop.width = Math.max(1, Math.round((rect[2] - rect[0]) * ratio));
      crop.height = Math.max(1, Math.round((rect[3] - rect[1]) * ratio));
      crop
        .getContext("2d")
        ?.drawImage(
          canvas,
          rect[0] * ratio,
          rect[1] * ratio,
          crop.width,
          crop.height,
          0,
          0,
          crop.width,
          crop.height,
        );
      setSelection({
        kind: "figure",
        page: pageIndex + 1,
        word_ids: [],
        rect,
        preview: `그림 · ${pageIndex + 1}쪽`,
        image: crop.toDataURL("image/png"),
      });
    } else {
      const target = window.document
        .elementFromPoint(event.clientX, event.clientY)
        ?.closest("[data-word]")
        ?.getAttribute("data-word");
      const first = words.findIndex((w) => w.id === origin.word),
        last = words.findIndex((w) => w.id === target);
      const range =
        first >= 0 && last >= 0
          ? words.slice(Math.min(first, last), Math.max(first, last) + 1)
          : selected;
      if (!range.length) {
        setBox(null);
        return;
      }
      setSelection({
        kind: "text",
        page: pageIndex + 1,
        word_ids: range.slice(0, 300).map((w) => w.id),
        preview: range
          .slice(0, 300)
          .map((w) => w.text)
          .join(" "),
      });
    }
    setBox(rect);
  }
  return (
    <>
      <div className="selection-toolbar" role="toolbar" aria-label="PDF 도구">
        {(
          [
            ["read", "단어 뜻"],
            ["text", "문장 선택"],
            ["figure", "그림 선택"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            aria-pressed={mode === value}
            onClick={() => {
              setMode(value);
              setSelection(null);
              setBox(null);
            }}
          >
            {label}
          </button>
        ))}
        <span>
          {mode === "read" ? "클릭해서 뜻 보기" : "영역을 드래그하세요"}
        </span>
      </div>
      {selection && (
        <div
          className="selection-preview"
          draggable
          onDragStart={(event) => {
            event.dataTransfer.setData(
              "application/readpaper-selection",
              JSON.stringify(selection),
            );
            event.dataTransfer.effectAllowed = "copy";
          }}
        >
          {selection.image && <img src={selection.image} alt="선택한 그림" />}
          <span>{selection.preview}</span>
          <button
            onClick={() => {
              onAttach(selection);
              setSelection(null);
              setBox(null);
              setMode("read");
            }}
          >
            채팅에 첨부 ↗
          </button>
          <button
            aria-label="선택 취소"
            onClick={() => {
              setSelection(null);
              setBox(null);
            }}
          >
            ×
          </button>
        </div>
      )}
      <div ref={frameRef} className="pdf-frame">
        <div
          className={`pdf-page pdf-page--${mode}`}
          style={{
            width: (page?.width || dimensions.width) * scale,
            height: (page?.height || dimensions.height) * scale,
          }}
          onPointerDown={(event) => {
            if (mode === "read") return;
            const p = point(event);
            start.current = {
              ...p,
              word:
                (event.target as HTMLElement)
                  .closest("[data-word]")
                  ?.getAttribute("data-word") || undefined,
            };
            setSelection(null);
            setBox(null);
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={(event) => {
            if (start.current) {
              const p = point(event),
                a = start.current;
              setBox([
                Math.min(a.x, p.x),
                Math.min(a.y, p.y),
                Math.max(a.x, p.x),
                Math.max(a.y, p.y),
              ]);
            }
          }}
          onPointerUp={makeSelection}
          onPointerCancel={() => {
            start.current = null;
            setBox(null);
          }}
        >
          <canvas
            ref={canvasRef}
            className="pdf-canvas"
            aria-label={`논문 ${pageIndex + 1}페이지`}
          />
          {words.map((word) => {
            const ref = byWord.get(word.id),
              pending = pendingWordIds.has(word.id);
            const active = Boolean(ref && translated.has(ref.key));
            return (
              <button
                key={word.id}
                data-word={word.id}
                type="button"
                className={`pdf-word ${pending ? "pdf-word--pending" : ""} ${selection?.word_ids.includes(word.id) ? "pdf-word--selected" : ""}`}
                style={{
                  left: word.x0 * scale,
                  top: word.y0 * scale,
                  width: (word.x1 - word.x0) * scale,
                  height: (word.y1 - word.y0) * scale,
                }}
                title={pending ? "번역 중…" : `${word.text} 뜻 보기`}
                aria-label={`${word.text} 뜻 보기`}
                disabled={pending && mode === "read"}
                onClick={() => {
                  if (mode === "read") onWord(word, ref?.gloss);
                }}
                onDoubleClick={() => {
                  if (mode === "text") {
                    const row = words.filter(
                      (w) => w.sentence_id === word.sentence_id,
                    );
                    setSelection({
                      kind: "text",
                      page: pageIndex + 1,
                      word_ids: row.map((w) => w.id),
                      preview: row.map((w) => w.text).join(" "),
                    });
                  }
                }}
                aria-pressed={active}
              />
            );
          })}
          {mode === "read" &&
            overlays.map(({ x, y, right, bottom, ref, label, line }, i) => {
              const width = (right - x) * scale,
                height = (bottom - y) * scale;
              const fontSize = Math.min(15, height * 0.82);
              const compressed = label.length * fontSize > width - 4;
              return (
                <button
                  key={`${ref.key}-${i}`}
                  className="pdf-gloss"
                  title={`${ref.gloss.original} → ${ref.gloss.meaning}`}
                  aria-label={`${ref.gloss.meaning}, 원문으로 전환`}
                  onClick={() => onWord(line[0], ref.gloss)}
                  style={{ left: x * scale, top: y * scale, width, height }}
                >
                  <svg width="100%" height="100%" aria-hidden="true">
                    <text
                      x="50%"
                      y="50%"
                      dy=".35em"
                      textAnchor="middle"
                      fontSize={fontSize}
                      textLength={
                        compressed ? Math.max(width - 4, 1) : undefined
                      }
                      lengthAdjust="spacingAndGlyphs"
                    >
                      {label}
                    </text>
                  </svg>
                </button>
              );
            })}
          {box && mode !== "read" && (
            <div
              className="pdf-selection-box"
              style={{
                left: box[0] * scale,
                top: box[1] * scale,
                width: (box[2] - box[0]) * scale,
                height: (box[3] - box[1]) * scale,
              }}
            />
          )}
        </div>
        {renderError && <p className="inline-error">{renderError}</p>}
      </div>
    </>
  );
}
