import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type DragEvent,
} from "react";
import * as pdfjs from "pdfjs-dist";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { api } from "./api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CitationGraph from "./CitationGraph";
import PdfReader from "./PdfReader";
import type {
  ActivityEvent,
  Answer,
  Attachment,
  GlossPreparation,
  Paper,
  Source,
  VocabularyItem,
  Word,
  WordGloss,
} from "./types";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.mjs",
  import.meta.url,
).toString();

type Panel = "assistant" | "research" | "vocabulary";
type ChatTurn = {
  id: number;
  question: string;
  answer?: Answer;
  pending?: boolean;
  attachments?: Attachment[];
  error?: string;
};

const sourceLabels: Record<Source["kind"], string> = {
  paper: "현재 논문",
  referenced: "참고 논문",
  related: "관련 연구",
  citing: "인용한 논문",
  recent: "최신 연구",
  background: "배경 자료",
};

function Icon({
  name,
  size = 18,
}: {
  name:
    | "upload"
    | "book"
    | "chat"
    | "network"
    | "words"
    | "send"
    | "spark"
    | "check"
    | "search"
    | "arrow";
  size?: number;
}) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true as const,
  };
  switch (name) {
    case "upload":
      return (
        <svg {...common}>
          <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
          <path d="M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5" />
        </svg>
      );
    case "book":
      return (
        <svg {...common}>
          <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 0 4 22V5.5Z" />
          <path d="M4 17a2.5 2.5 0 0 1 2.5-2.5H20M8 7h8" />
        </svg>
      );
    case "chat":
      return (
        <svg {...common}>
          <path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H5l-1 2v-6.5A7.5 7.5 0 1 1 20 11.5Z" />
          <path d="M8 11h8M8 14h5" />
        </svg>
      );
    case "network":
      return (
        <svg {...common}>
          <circle cx="12" cy="5" r="2.5" />
          <circle cx="5" cy="18" r="2.5" />
          <circle cx="19" cy="18" r="2.5" />
          <path d="m10.8 7.2-4.5 8.6m6.9-8.6 4.5 8.6M7.5 18h9" />
        </svg>
      );
    case "words":
      return (
        <svg {...common}>
          <path d="M4 6h16M4 12h10M4 18h7" />
          <path d="m17 15 3 6m-6 0 3-6m-2.1 4h4.2" />
        </svg>
      );
    case "send":
      return (
        <svg {...common}>
          <path d="m21 3-7.2 18-3.6-7.2L3 10.2 21 3Z" />
          <path d="M10.2 13.8 15 9" />
        </svg>
      );
    case "spark":
      return (
        <svg {...common}>
          <path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3Z" />
          <path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" />
        </svg>
      );
    case "check":
      return (
        <svg {...common}>
          <path d="m5 12 4 4L19 6" />
        </svg>
      );
    case "search":
      return (
        <svg {...common}>
          <circle cx="10.8" cy="10.8" r="6.8" />
          <path d="m16 16 4.5 4.5" />
        </svg>
      );
    case "arrow":
      return (
        <svg {...common}>
          <path d="M5 12h14m-6-6 6 6-6 6" />
        </svg>
      );
  }
}

function Brand() {
  return (
    <div className="brand">
      <span className="brand-mark">r</span>
      <span>readpaper</span>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState<string | null | undefined>(undefined);
  const [loginName, setLoginName] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [paperId, setPaperId] = useState<string | null>(null);
  const [paper, setPaper] = useState<Paper | null>(null);
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [fileName, setFileName] = useState("");
  const [pageIndex, setPageIndex] = useState(0);
  const [glossPreparation, setGlossPreparation] =
    useState<GlossPreparation | null>(null);
  const [skipPreparation, setSkipPreparation] = useState(false);
  const [preparationAttempt, setPreparationAttempt] = useState(0);
  const [panel, setPanel] = useState<Panel>("assistant");
  const [gloss, setGloss] = useState<{
    original: string;
    meaning: string;
    sentence: string;
    contextSource?: WordGloss["context_source"];
    elapsedMs?: number;
  } | null>(null);
  const [translated, setTranslated] = useState<Set<string>>(new Set());
  const [pendingWordIds, setPendingWordIds] = useState<Set<string>>(new Set());
  const [vocabulary, setVocabulary] = useState<VocabularyItem[]>([]);
  const [question, setQuestion] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [asking, setAsking] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState("");
  const [manualTerm, setManualTerm] = useState("");
  const [manualMeaning, setManualMeaning] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const chatScroll = useRef<HTMLDivElement>(null);
  const uploadGeneration = useRef(0);
  const wordRequests = useRef(new Map<string, Promise<WordGloss>>());

  useEffect(() => {
    api
      .me()
      .then((result) => setUser(result.username))
      .catch(() => setUser(null));
  }, []);

  const refreshVocabulary = useCallback(() => {
    void api
      .vocabulary()
      .then(setVocabulary)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (user) refreshVocabulary();
  }, [user, refreshVocabulary]);

  useEffect(() => {
    if (!paperId || paper?.status === "ready" || paper?.status === "failed")
      return;
    let active = true;
    const poll = () => {
      void api
        .paper(paperId)
        .then((result) => {
          if (active)
            setPaper((previous) =>
              previous?.id === result.id
                ? {
                    ...result,
                    word_glosses: {
                      ...previous.word_glosses,
                      ...result.word_glosses,
                    },
                  }
                : result,
            );
        })
        .catch((error: Error) => {
          if (active) setNotice(error.message);
        });
    };
    poll();
    const timer = window.setInterval(poll, 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [paperId, paper?.status]);

  useEffect(() => {
    if (!paperId) return;
    let active = true;
    let timer: number | undefined;
    const poll = () => {
      void api
        .glossPreparation(paperId)
        .then((result) => {
          if (!active) return;
          setGlossPreparation(result);
          setPaper((previous) =>
            previous?.id === paperId
              ? {
                  ...previous,
                  word_glosses: {
                    ...previous.word_glosses,
                    ...result.word_glosses,
                  },
                }
              : previous,
          );
          if (result.status === "completed" || result.status === "failed")
            window.clearInterval(timer);
        })
        .catch((error: Error) => {
          if (!active) return;
          window.clearInterval(timer);
          setGlossPreparation({
            status: "failed",
            total: 0,
            ready: 0,
            failed: 0,
            stage: "failed",
            error: error.message,
            summary: null,
            word_glosses: {},
          });
        });
    };
    poll();
    timer = window.setInterval(poll, 1500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [paperId, preparationAttempt]);

  useEffect(() => {
    chatScroll.current?.scrollTo({
      top: chatScroll.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, asking]);

  const sentenceMap = useMemo(
    () =>
      new Map(
        (paper?.sentences || []).map((sentence) => [sentence.id, sentence]),
      ),
    [paper?.sentences],
  );
  const currentPage = paper?.pages?.[pageIndex];
  const canAsk = Boolean(
    paperId &&
    paper?.status === "ready" &&
    paper?.rag_index?.status === "ready",
  );

  async function doLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setNotice("");
    try {
      const result = await api.login(loginName, loginPassword);
      setUser(result.username);
      setLoginPassword("");
    } catch (error) {
      setNotice((error as Error).message);
    }
  }

  async function upload(file: File) {
    uploadGeneration.current += 1;
    setAsking(false);
    setNotice("");
    setUploading(true);
    setPdf(null);
    setFileName("");
    setPaper(null);
    setPaperId(null);
    setTranslated(new Set());
    setPendingWordIds(new Set());
    wordRequests.current.clear();
    setGloss(null);
    setTurns([]);
    setQuestion("");
    setAttachments([]);
    setPanel("assistant");
    setPageIndex(0);
    setGlossPreparation(null);
    setSkipPreparation(false);
    try {
      if (file.size > 25 * 1024 * 1024)
        throw new Error("PDF는 25MB 이하로 업로드해 주세요.");
      const bytes = new Uint8Array(await file.arrayBuffer());
      const loaded = await pdfjs.getDocument({ data: bytes }).promise;
      setPdf(loaded);
      setFileName(file.name);
      const result = await api.upload(file);
      setPaperId(result.paper_id);
      setPaper(await api.paper(result.paper_id));
      if (result.cached) setNotice("이전에 분석한 논문을 불러왔습니다.");
    } catch (error) {
      setPdf(null);
      setPaper(null);
      setPaperId(null);
      setNotice((error as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function retryResearch() {
    if (!paperId) return;
    try {
      await api.retry(paperId);
      setPaper((previous) =>
        previous
          ? { ...previous, status: "processing", progress: 0, error: null }
          : previous,
      );
    } catch (error) {
      setNotice((error as Error).message);
    }
  }

  async function retryPreparation() {
    if (!paperId) return;
    try {
      const result = await api.retryGlossPreparation(paperId);
      setGlossPreparation(result);
      setPreparationAttempt((previous) => previous + 1);
    } catch (error) {
      setNotice((error as Error).message);
    }
  }

  function revealGloss(unit: WordGloss, word: Word, elapsedMs?: number) {
    const key = unit.word_ids.join(":");
    const wasTranslated = translated.has(key);
    setTranslated((previous) => {
      const next = new Set(previous);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
    if (wasTranslated) {
      setGloss(null);
      return;
    }
    const sentence = sentenceMap.get(unit.sentence_id || word.sentence_id);
    setGloss({
      original: unit.original,
      meaning: unit.meaning,
      sentence: sentence?.text || "",
      contextSource: unit.context_source,
      elapsedMs,
    });
    if (paper) {
      void api
        .saveWord({
          paper_title: paper.title,
          term: unit.original,
          meaning: unit.meaning,
          source_sentence: sentence?.text || "",
        })
        .then(refreshVocabulary)
        .catch((error: Error) => setNotice(`단어 저장 실패: ${error.message}`));
    }
  }

  function toggleWord(word: Word, cached?: WordGloss) {
    if (cached) {
      revealGloss(cached, word);
      return;
    }
    if (!paperId || wordRequests.current.has(word.id)) return;

    const generation = uploadGeneration.current;
    const clickedAt = performance.now();
    setPendingWordIds((previous) => new Set(previous).add(word.id));
    const request = api
      .translateWord(paperId, word.id)
      .then((result) => {
        const unit: WordGloss = {
          word_ids: result.word_ids,
          original: result.original,
          meaning: result.meaning,
          sentence_id: result.sentence_id,
          context_source: result.context_source,
        };
        if (generation !== uploadGeneration.current) return unit;
        setPaper((previous) =>
          previous
            ? {
                ...previous,
                activity: result.activity || previous.activity,
                word_glosses: {
                  ...previous.word_glosses,
                  ...Object.fromEntries(unit.word_ids.map((id) => [id, unit])),
                },
              }
            : previous,
        );
        revealGloss(unit, word, performance.now() - clickedAt);
        return unit;
      })
      .catch((error: Error) => {
        if (generation === uploadGeneration.current) setNotice(error.message);
        throw error;
      })
      .finally(() => {
        if (generation !== uploadGeneration.current) return;
        wordRequests.current.delete(word.id);
        setPendingWordIds((previous) => {
          const next = new Set(previous);
          next.delete(word.id);
          return next;
        });
      });
    wordRequests.current.set(word.id, request);
    void request.catch(() => undefined);
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = question.trim();
    if (!paperId || !prompt || !canAsk || asking) return;
    const generation = uploadGeneration.current;
    const id = Date.now();
    const sentAttachments = attachments;
    const history = turns
      .filter((t) => t.answer)
      .flatMap((t) => [
        { role: "user" as const, content: t.question.slice(0, 2000) },
        {
          role: "assistant" as const,
          content: t.answer!.answer.slice(0, 2000),
        },
      ])
      .slice(-6);
    setTurns((previous) => [
      ...previous,
      { id, question: prompt, attachments: sentAttachments, pending: true },
    ]);
    setAttachments([]);
    setQuestion("");
    setAsking(true);
    setNotice("");
    try {
      const result = await api.ask(paperId, prompt, sentAttachments, history);
      if (generation !== uploadGeneration.current) return;
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === id ? { ...turn, answer: result, pending: false } : turn,
        ),
      );
    } catch (error) {
      if (generation !== uploadGeneration.current) return;
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === id
            ? { ...turn, pending: false, error: (error as Error).message }
            : turn,
        ),
      );
      setNotice((error as Error).message);
    } finally {
      if (generation === uploadGeneration.current) setAsking(false);
    }
  }

  async function saveManual(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      await api.saveWord({
        paper_title: paper?.title || "직접 추가",
        term: manualTerm,
        meaning: manualMeaning,
        source_sentence: "",
      });
      setManualTerm("");
      setManualMeaning("");
      refreshVocabulary();
    } catch (error) {
      setNotice((error as Error).message);
    }
  }

  function attach(item: Attachment) {
    if (attachments.length >= 4) {
      setNotice("한 번에 4개까지 첨부할 수 있습니다.");
      return;
    }
    setAttachments((previous) => [...previous, item]);
    setPanel("assistant");
  }

  function handleDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void upload(file);
  }

  if (user === undefined)
    return (
      <div className="boot-screen">
        <span className="boot-mark">r</span>
        <span>readpaper</span>
      </div>
    );

  if (!user)
    return (
      <div className="login-page">
        <div className="login-card">
          <Brand />
          <p className="eyebrow">A QUIETER WAY TO READ</p>
          <h1>
            논문을 읽는
            <br />
            <em>나만의 자리</em>
          </h1>
          <p className="muted">
            읽다가 막히는 순간, 맥락을 잃지 않고 뜻을 확인하세요.
          </p>
          <form onSubmit={doLogin} className="login-form">
            <label>
              아이디
              <input
                value={loginName}
                onChange={(event) => setLoginName(event.target.value)}
                autoComplete="username"
                required
              />
            </label>
            <label>
              비밀번호
              <input
                type="password"
                value={loginPassword}
                onChange={(event) => setLoginPassword(event.target.value)}
                autoComplete="current-password"
                required
              />
            </label>
            <button className="primary-button" type="submit">
              읽기 시작하기 <Icon name="arrow" size={16} />
            </button>
          </form>
          {notice && <p className="inline-error">{notice}</p>}
        </div>
        <div className="login-side">
          <span className="login-index">READPAPER / 01</span>
          <p>
            Good research
            <br />
            takes <em>time.</em>
          </p>
          <span className="login-caption">TAKE YOURS.</span>
        </div>
      </div>
    );

  const sourceCounts = countSources(paper?.sources || []);
  const preparingSummary = Boolean(
    pdf &&
    !skipPreparation &&
    (uploading ||
      !paper ||
      glossPreparation?.status !== "completed" ||
      paper.status !== "ready"),
  );

  return (
    <div className="app-shell">
      <header className="site-header">
        <Brand />
        <div className="header-right">
          {paper?.status === "ready" && !preparingSummary && (
            <span className="header-state">
              <span className="status-dot" /> 읽을 준비가 됐어요
            </span>
          )}
          <span className="user-name">{user}</span>
          <button
            className="text-button"
            onClick={() => {
              void api.logout().then(() => {
                setUser(null);
                setPaper(null);
                setPaperId(null);
                setPdf(null);
                setFileName("");
              });
            }}
          >
            로그아웃
          </button>
        </div>
      </header>

      {!pdf ? (
        <main className="welcome-page">
          <section className="welcome-copy">
            <h1>
              읽고, 묻고,
              <br />
              이해하세요.
            </h1>
            <p className="welcome-description">
              논문 한 편에서 시작하는 나만의 연구 공간.
            </p>
          </section>
          <section
            className={`upload-card ${dragging ? "upload-card--dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div className="upload-card-top">
              <span>START WITH A PAPER</span>
              <span>PDF / UP TO 25 MB</span>
            </div>
            <div className="upload-illustration" aria-hidden="true">
              <div className="upload-sheet">
                <span />
                <span />
                <span />
                <span />
              </div>
              <div className="upload-arrow">
                <Icon name="upload" size={21} />
              </div>
            </div>
            <h2>
              읽고 싶은 논문을
              <br />
              여기에 놓아주세요.
            </h2>
            <p>텍스트를 선택할 수 있는 PDF 파일을 업로드합니다.</p>
            <button
              className="primary-button"
              disabled={uploading}
              onClick={() => fileInput.current?.click()}
            >
              {uploading ? "PDF를 여는 중…" : "PDF 파일 선택"}
              <Icon name="arrow" size={16} />
            </button>
            <div className="drop-hint">또는 파일을 이 영역으로 끌어오세요</div>
          </section>
          <input
            ref={fileInput}
            hidden
            type="file"
            accept="application/pdf,.pdf"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void upload(file);
              event.currentTarget.value = "";
            }}
          />

          {notice && (
            <div className="toast" role="status">
              {notice}
              <button onClick={() => setNotice("")} aria-label="닫기">
                ×
              </button>
            </div>
          )}
        </main>
      ) : preparingSummary ? (
        <main className="preparation-page" aria-live="polite">
          <div className="preparation-card">
            <p className="eyebrow">논문 준비</p>
            <h1>
              {paper?.status === "failed" ||
              glossPreparation?.status === "failed"
                ? "준비를 완료하지 못했습니다."
                : glossPreparation?.stage === "summarizing"
                  ? "논문의 핵심을 읽고 있어요."
                  : "연결된 연구를 준비하고 있어요."}
            </h1>
            <p className="preparation-title">{paper?.title || fileName}</p>
            {glossPreparation?.status !== "failed" &&
              paper?.status !== "failed" && (
                <div
                  className="preparation-progress"
                  role="progressbar"
                  aria-label="논문 요약 준비 중"
                >
                  <span />
                </div>
              )}
            {glossPreparation?.summary && (
              <p className="preparation-summary">{glossPreparation.summary}</p>
            )}
            <p className="preparation-note">
              인용·관련 논문 수집 → 본문 읽기 → 검색 준비
            </p>
            <ActivityLog
              activity={paper?.activity || []}
              live={paper?.status === "processing"}
            />
            {paper?.status === "failed" && (
              <div className="preparation-actions">
                <p>{paper.error}</p>
                <button
                  className="primary-button"
                  onClick={() => {
                    if (paperId)
                      void api
                        .retry(paperId)
                        .then(() =>
                          setPaper((previous) =>
                            previous
                              ? {
                                  ...previous,
                                  status: "processing",
                                  error: null,
                                }
                              : previous,
                          ),
                        )
                        .catch((error) => setNotice(error.message));
                  }}
                >
                  문헌 준비 다시 시도
                </button>
              </div>
            )}
            <button
              className="text-button"
              onClick={() => setSkipPreparation(true)}
            >
              원문 먼저 열기 ↗
            </button>
            {glossPreparation?.status === "failed" && (
              <div className="preparation-actions">
                <p>
                  {glossPreparation.error || "로컬 모델 연결을 확인해 주세요."}
                </p>
                <button
                  className="primary-button"
                  onClick={() => {
                    void retryPreparation();
                  }}
                >
                  다시 시도
                </button>
                <button
                  className="secondary-button"
                  onClick={() => setSkipPreparation(true)}
                >
                  요약 없이 원문 먼저 읽기
                </button>
              </div>
            )}
          </div>
        </main>
      ) : (
        <main className="workspace">
          <div className="paper-toolbar">
            <div className="paper-heading-icon">
              <Icon name="book" size={20} />
            </div>
            <div className="paper-title-block">
              <h1 title={paper?.title}>{paper?.title || fileName}</h1>
              <span>{fileName}</span>
            </div>
            <div className="toolbar-actions">
              {paper?.status === "ready" && (
                <span className="ready-label">
                  <span className="status-dot" /> READY
                </span>
              )}
              <button
                className="secondary-button"
                onClick={() => fileInput.current?.click()}
              >
                <span>＋</span> 새 논문
              </button>
            </div>
            <input
              ref={fileInput}
              hidden
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void upload(file);
                event.currentTarget.value = "";
              }}
            />
          </div>

          <AnalysisTimeline
            paper={paper}
            paperId={paperId}
            uploading={uploading}
            onRetry={() => {
              if (!paperId || !paper) return;
              void api
                .retry(paperId)
                .then(() =>
                  setPaper({
                    ...paper,
                    status: "processing",
                    progress: 0,
                    error: null,
                  }),
                )
                .catch((error: Error) => setNotice(error.message));
            }}
          />

          <div className="reader-layout">
            <section className="reader-main" aria-label="논문 읽기">
              <div className="reader-controls">
                <div className="reader-label">
                  <Icon name="book" size={16} /> 원문
                </div>
                <div className="page-nav">
                  <button
                    disabled={pageIndex === 0}
                    onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
                    aria-label="이전 페이지"
                  >
                    ‹
                  </button>
                  <span>
                    {pageIndex + 1}
                    <i>/</i>
                    {pdf.numPages}
                  </span>
                  <button
                    disabled={pageIndex + 1 >= pdf.numPages}
                    onClick={() =>
                      setPageIndex(Math.min(pdf.numPages - 1, pageIndex + 1))
                    }
                    aria-label="다음 페이지"
                  >
                    ›
                  </button>
                </div>
              </div>

              <div className="pdf-stage">
                <PdfReader
                  document={pdf}
                  pageIndex={pageIndex}
                  page={currentPage}
                  wordGlosses={{
                    ...paper?.word_glosses,
                    ...glossPreparation?.word_glosses,
                  }}
                  translated={translated}
                  pendingWordIds={pendingWordIds}
                  onWord={toggleWord}
                  onAttach={attach}
                />
              </div>
              <div className="reader-foot">
                <span>
                  <Icon name="book" size={14} />{" "}
                  {paper?.pages?.length || pdf.numPages} PAGES
                </span>
                <span>클릭한 단어는 내 단어장에 저장됩니다</span>
              </div>
            </section>

            <aside className="research-dock" aria-label="논문 도구">
              <div className="dock-header">
                <div>
                  <h2>논문에 질문하기</h2>
                </div>
                <span className="dock-status">
                  <i />{" "}
                  {paper?.rag_index
                    ? `${paper.rag_index.chunks}개 근거`
                    : "검색 준비 중"}
                </span>
              </div>
              <div className="dock-tabs" role="tablist" aria-label="읽기 도구">
                <button
                  role="tab"
                  aria-selected={panel === "assistant"}
                  className={panel === "assistant" ? "active" : ""}
                  onClick={() => setPanel("assistant")}
                >
                  <Icon name="chat" size={16} />
                  <span>대화</span>
                </button>
                <button
                  role="tab"
                  aria-selected={panel === "research"}
                  className={panel === "research" ? "active" : ""}
                  onClick={() => setPanel("research")}
                >
                  <Icon name="network" size={16} />
                  <span>인용 지도</span>
                  {sourceCounts.references + sourceCounts.citing > 0 && (
                    <b>{sourceCounts.references + sourceCounts.citing}</b>
                  )}
                </button>
                <button
                  role="tab"
                  aria-selected={panel === "vocabulary"}
                  className={panel === "vocabulary" ? "active" : ""}
                  onClick={() => setPanel("vocabulary")}
                >
                  <Icon name="words" size={16} />
                  <span>단어장</span>
                </button>
              </div>

              {panel === "assistant" && (
                <section
                  className="dock-panel assistant-panel"
                  role="tabpanel"
                  onDragOver={(event) => {
                    if (
                      event.dataTransfer.types.includes(
                        "application/readpaper-selection",
                      )
                    )
                      event.preventDefault();
                  }}
                  onDrop={(event) => {
                    const raw = event.dataTransfer.getData(
                      "application/readpaper-selection",
                    );
                    if (raw) {
                      event.preventDefault();
                      try {
                        const value = JSON.parse(raw);
                        if (
                          ["text", "figure"].includes(value.kind) &&
                          Array.isArray(value.word_ids) &&
                          typeof value.preview === "string"
                        )
                          attach(value);
                      } catch {
                        setNotice("선택 내용을 다시 첨부해 주세요.");
                      }
                    }
                  }}
                >
                  {gloss && (
                    <div className="gloss-card">
                      <div className="gloss-term">
                        <strong>{gloss.original}</strong>
                        <span>{gloss.meaning}</span>
                        <button
                          onClick={() => setGloss(null)}
                          aria-label="뜻 닫기"
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  )}
                  <div className="chat-scroll" ref={chatScroll}>
                    {turns.length === 0 ? (
                      <div className="chat-empty">
                        <div className="assistant-avatar">
                          <Icon name="chat" size={24} />
                        </div>
                        <p className="chat-greeting">어떤 점이 궁금한가요?</p>
                        <p>문장이나 그림을 선택해 함께 물어보세요.</p>
                        <div className="prompt-list">
                          {[
                            "이 논문의 핵심 기여는 무엇인가요?",
                            "기존 연구와 어떤 차이가 있나요?",
                            "핵심 방법을 쉽게 설명해줘",
                          ].map((prompt) => (
                            <button
                              key={prompt}
                              onClick={() => setQuestion(prompt)}
                            >
                              {prompt}
                              <Icon name="arrow" size={16} />
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="chat-turns">
                        {turns.map((turn) => (
                          <div className="chat-turn" key={turn.id}>
                            <div className="user-bubble">
                              {turn.attachments?.map((item, i) => (
                                <div className="sent-attachment" key={i}>
                                  {item.image && (
                                    <img
                                      src={item.image}
                                      alt="질문에 첨부한 그림"
                                    />
                                  )}
                                  <small>
                                    {item.page}쪽 · {item.preview}
                                  </small>
                                </div>
                              ))}
                              {turn.question}
                            </div>
                            {turn.pending ? (
                              <div className="thinking-line" role="status">
                                <span />
                                <span />
                                <span /> 근거를 확인하고 있어요
                              </div>
                            ) : turn.answer ? (
                              <div className="assistant-bubble">
                                <ReactMarkdown
                                  remarkPlugins={[remarkGfm]}
                                  components={{
                                    a: ({ href, children }) => (
                                      <a
                                        href={href}
                                        target={
                                          href?.startsWith("#")
                                            ? undefined
                                            : "_blank"
                                        }
                                        rel="noopener noreferrer"
                                        onClick={() => {
                                          if (href?.startsWith("#evidence-")) {
                                            const node =
                                              document.getElementById(
                                                href.slice(1),
                                              );
                                            const details =
                                              node?.closest("details");
                                            if (details) details.open = true;
                                          }
                                        }}
                                      >
                                        {children}
                                      </a>
                                    ),
                                  }}
                                >
                                  {turn.answer.answer.replace(
                                    /\[([^\]]+)\]/g,
                                    (whole, id: string) => {
                                      const index =
                                        turn.answer!.sources.findIndex(
                                          (source) => source.id === id,
                                        );
                                      return index < 0
                                        ? whole
                                        : `[${index + 1}](#evidence-${turn.id}-${index})`;
                                    },
                                  )}
                                </ReactMarkdown>
                                {turn.answer.sources.length > 0 && (
                                  <details className="answer-sources">
                                    <summary>
                                      출처 {turn.answer.sources.length}개
                                    </summary>
                                    {turn.answer.sources.map((source, i) => (
                                      <div
                                        id={`evidence-${turn.id}-${i}`}
                                        key={source.id}
                                      >
                                        <SourceCard
                                          source={source}
                                          compact
                                          excerpt
                                        />
                                      </div>
                                    ))}
                                  </details>
                                )}
                              </div>
                            ) : (
                              <div className="chat-error" role="alert">
                                {turn.error || "답변을 받지 못했습니다."}
                                <button
                                  onClick={() => {
                                    setQuestion(turn.question);
                                    setAttachments(turn.attachments || []);
                                  }}
                                >
                                  질문 다시 가져오기
                                </button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <form className="chat-form" onSubmit={ask}>
                    {attachments.length > 0 && (
                      <div className="attachment-list">
                        {attachments.map((item, i) => (
                          <div className="attachment-chip" key={i}>
                            {item.image && (
                              <img src={item.image} alt="첨부할 그림" />
                            )}
                            <span>
                              <small>
                                {item.page}쪽 ·{" "}
                                {item.kind === "figure" ? "그림" : "선택 문장"}
                              </small>
                              {item.preview}
                            </span>
                            <button
                              type="button"
                              aria-label="첨부 삭제"
                              onClick={() =>
                                setAttachments((previous) =>
                                  previous.filter((_, index) => index !== i),
                                )
                              }
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    {!canAsk && (
                      <p className="chat-disabled-note">
                        문헌 검색을 준비하고 있습니다.
                      </p>
                    )}
                    <div className="chat-input-wrap">
                      <textarea
                        rows={2}
                        maxLength={1000}
                        value={question}
                        onChange={(event) => setQuestion(event.target.value)}
                        placeholder="논문에 대해 물어보세요…"
                        aria-label="질문"
                        disabled={asking}
                        onKeyDown={(event) => {
                          if (
                            event.key === "Enter" &&
                            !event.shiftKey &&
                            !event.nativeEvent.isComposing
                          ) {
                            event.preventDefault();
                            event.currentTarget.form?.requestSubmit();
                          }
                        }}
                      />
                      <button
                        type="submit"
                        disabled={!canAsk || asking || !question.trim()}
                        aria-label="질문 보내기"
                      >
                        <Icon name="arrow" size={20} />
                      </button>
                    </div>
                    <div className="chat-disclaimer">
                      선택 영역을 이곳으로 끌어와 첨부 · Shift + Enter 줄바꿈
                    </div>
                  </form>
                </section>
              )}

              {panel === "research" && (
                <ResearchPanel
                  title={paper?.title || fileName}
                  sources={paper?.sources || []}
                  warnings={paper?.source_warnings || []}
                  status={paper?.status || "processing"}
                  sourcesReady={Boolean(paper?.sources_ready)}
                  error={paper?.error || undefined}
                  onRetry={() => {
                    void retryResearch();
                  }}
                />
              )}
              {panel === "vocabulary" && (
                <div className="dock-panel vocabulary-panel" role="tabpanel">
                  <div className="panel-title-row">
                    <div>
                      <p className="eyebrow">YOUR READING NOTES</p>
                      <h2>
                        나의 단어장{" "}
                        <span className="count">{vocabulary.length}</span>
                      </h2>
                    </div>
                  </div>
                  <p className="panel-copy">
                    논문을 읽으며 확인한 단어가 모입니다.
                  </p>
                  <form className="manual-form" onSubmit={saveManual}>
                    <input
                      value={manualTerm}
                      onChange={(event) => setManualTerm(event.target.value)}
                      placeholder="단어"
                      required
                    />
                    <input
                      value={manualMeaning}
                      onChange={(event) => setManualMeaning(event.target.value)}
                      placeholder="뜻"
                      required
                    />
                    <button className="secondary-button" type="submit">
                      추가
                    </button>
                  </form>
                  <div className="vocab-list">
                    {vocabulary.map((item) => (
                      <article className="vocab-item" key={item.id}>
                        <div>
                          <strong>{item.term}</strong>
                          <span>{item.meaning}</span>
                        </div>
                        {item.source_sentence && <p>{item.source_sentence}</p>}
                        <small>{item.paper_title}</small>
                        <button
                          aria-label={`${item.term} 삭제`}
                          onClick={() => {
                            void api
                              .deleteWord(item.id)
                              .then(refreshVocabulary)
                              .catch((error: Error) =>
                                setNotice(error.message),
                              );
                          }}
                        >
                          ×
                        </button>
                      </article>
                    ))}
                    {vocabulary.length === 0 && (
                      <div className="empty-research">
                        <span>
                          <Icon name="words" size={24} />
                        </span>
                        <p>논문에서 모르는 단어를 눌러보세요.</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </aside>
          </div>
          {notice && (
            <div className="toast" role="status">
              {notice}
              <button onClick={() => setNotice("")} aria-label="닫기">
                ×
              </button>
            </div>
          )}
        </main>
      )}
    </div>
  );
}

function countSources(sources: Source[]) {
  return {
    references: sources.filter((source) => source.kind === "referenced").length,
    citing: sources.filter((source) => source.kind === "citing").length,
    related: sources.filter(
      (source) => source.kind === "related" || source.kind === "recent",
    ).length,
  };
}

function AnalysisTimeline({
  paper,
  paperId,
  uploading,
  onRetry,
}: {
  paper: Paper | null;
  paperId: string | null;
  uploading: boolean;
  onRetry: () => void;
}) {
  const status = paper?.status;
  const activity = paper?.activity || [];
  const counts = countSources(paper?.sources || []);
  return (
    <details className="analysis-timeline">
      <summary>
        <span
          className={`status-dot ${status === "processing" ? "processing" : ""}`}
        />
        <strong>
          {status === "ready"
            ? "연구 자료 준비 완료"
            : status === "failed"
              ? "문헌 준비 실패"
              : "연구 자료를 수집하고 있어요"}
        </strong>
        <span>
          {counts.references}편 인용 · {counts.related}편 관련
          {paper?.rag_index ? ` · ${paper.rag_index.chunks}개 청크` : ""}
        </span>
        <span className="log-toggle">진행 기록 ⌄</span>
      </summary>
      <ActivityLog
        activity={activity}
        live={uploading || status === "processing"}
      />
      {paper?.error && <p className="inline-error">{paper.error}</p>}
      {status === "failed" && paperId && (
        <button className="secondary-button" onClick={onRetry}>
          다시 시도
        </button>
      )}
    </details>
  );
}

function ActivityLog({
  activity,
  live,
}: {
  activity: ActivityEvent[];
  live: boolean;
}) {
  const entries = [...activity].reverse();
  return (
    <section className="timeline-activity" aria-label="실시간 분석 로그">
      <header className="timeline-activity-heading">
        <div>
          <strong>수집·분석 로그</strong>
        </div>
        <span className={`activity-live ${live ? "activity-live--on" : ""}`}>
          <i />
          {live ? "LIVE" : "기록"}
        </span>
      </header>
      <div
        className="timeline-activity-list"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
        {entries.length > 0 ? (
          entries.map((event) => (
            <div
              className={`activity-entry activity-entry--${event.state}`}
              key={event.id}
            >
              <time dateTime={new Date(event.created_at * 1000).toISOString()}>
                {new Date(event.created_at * 1000).toLocaleTimeString("ko-KR", {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                  hour12: false,
                })}
              </time>
              <i aria-hidden="true" />
              <span>{event.message}</span>
            </div>
          ))
        ) : (
          <p className="activity-empty">
            PDF를 읽고 있습니다. 첫 처리 로그를 기다리는 중입니다.
          </p>
        )}
      </div>
    </section>
  );
}

function ResearchPanel({
  title,
  sources,
  warnings,
  status,
  sourcesReady,
  error,
  onRetry,
}: {
  title: string;
  sources: Source[];
  warnings: string[];
  status: Paper["status"];
  sourcesReady: boolean;
  error?: string;
  onRetry: () => void;
}) {
  const [query, setQuery] = useState("");
  const [graphExpanded, setGraphExpanded] = useState(false);
  const counts = countSources(sources);
  const citationsReady = sourcesReady || status === "ready";
  const related = sources.filter(
    (source) => source.kind === "related" || source.kind === "recent",
  );
  const searchable = sources.filter((source) => source.kind !== "background");
  const filtered = searchable.filter((source) =>
    `${source.title} ${(source.authors || []).join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <div className="dock-panel research-panel" role="tabpanel">
      <div className="research-panel-heading">
        <p className="eyebrow">PAPER CONNECTIONS</p>
        <h2>인용 지도</h2>
        <button
          className="secondary-button"
          disabled={status === "processing"}
          onClick={onRetry}
        >
          문헌 다시 수집
        </button>
      </div>
      {!citationsReady ? (
        <div
          className={`graph-wait ${status === "failed" ? "graph-wait--failed" : ""}`}
        >
          <div className="graph-wait-mark">
            <Icon name="network" size={26} />
          </div>
          <strong>
            {status === "failed"
              ? "문헌 관계를 가져오지 못했습니다"
              : "관계를 찾고 있습니다"}
          </strong>
          <p>
            {status === "failed"
              ? error || "검색을 완료하지 못했습니다. 다시 시도해 주세요."
              : "분석이 끝나면 참고문헌과 피인용 논문을 그래프로 보여드릴게요."}
          </p>
        </div>
      ) : (
        <>
          {status === "processing" && (
            <p className="collection-ready-note">
              문헌 검색이 끝나면 논문 문맥 검색을 준비합니다.
            </p>
          )}
          {status === "failed" && error && (
            <p className="source-warning">
              문헌 정보는 보존했습니다. 이후 분석 단계에서 오류가 발생했습니다:{" "}
              {error}
            </p>
          )}
          <div className="graph-counts">
            <div>
              <strong>{counts.references}</strong>
              <span>찾은 참고문헌</span>
            </div>
            <div>
              <strong>{counts.citing}</strong>
              <span>찾은 피인용</span>
            </div>
            <div>
              <strong>{counts.related}</strong>
              <span>관련 연구 후보</span>
            </div>
          </div>
          <div
            className={`citation-graph-card ${graphExpanded ? "citation-graph-card--expanded" : ""}`}
          >
            <button
              className="graph-expand secondary-button"
              onClick={() => setGraphExpanded(!graphExpanded)}
            >
              {graphExpanded ? "닫기 ×" : "크게 보기 ↗"}
            </button>
            <div className="graph-card-heading">
              <div>
                <strong>직접 인용 관계</strong>
                <span>화살표는 인용 방향을 나타냅니다</span>
              </div>
              <div className="graph-legend">
                <i className="legend-reference" />
                인용
                <i className="legend-citing" />
                피인용
              </div>
            </div>
            <CitationGraph title={title} sources={sources} />
          </div>
          <div className="research-results-heading">
            <div>
              <p className="eyebrow">DISCOVERED PAPERS</p>
              <h3>
                찾은 연구 <span>{searchable.length}</span>
              </h3>
            </div>
            <label className="source-search">
              <Icon name="search" size={15} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="제목으로 검색"
              />
            </label>
          </div>
          {warnings.map((warning) => (
            <p className="source-warning" key={warning}>
              {warning}
            </p>
          ))}
          {related.length > 0 && (
            <div className="related-note">
              <strong>관련 연구 후보</strong>
              <span>
                OpenAlex의 관련도 순서에서 최대 8편을 표시합니다. 인용 기록이
                확인된 연결만 그래프에 표시합니다.
              </span>
            </div>
          )}
          <div className="source-list">
            {filtered.map((source) => (
              <SourceCard key={source.id} source={source} compact />
            ))}
            {filtered.length === 0 && (
              <p className="empty-note">검색 결과가 없습니다.</p>
            )}
          </div>
          <p className="source-footnote">
            전문 미확보 자료는 초록 범위에서만 검색합니다.
          </p>
        </>
      )}
    </div>
  );
}

function SourceCard({
  source,
  excerpt = false,
  compact = false,
}: {
  source: Source;
  excerpt?: boolean;
  compact?: boolean;
}) {
  const scope =
    source.content_scope === "full_text"
      ? "전문"
      : source.content_scope === "abstract"
        ? "초록"
        : source.content_scope === "metadata"
          ? "서지 정보"
          : null;
  return (
    <article className={`source-card ${compact ? "source-card--compact" : ""}`}>
      <div className="source-card-top">
        <span className={`source-kind source-kind--${source.kind}`}>
          {sourceLabels[source.kind]}
        </span>
        {scope && <span>{scope}</span>}
        {source.year && <span>{source.year}</span>}
        {source.page && <span>{source.page}쪽</span>}
      </div>
      <h3>
        {source.url ? (
          <a href={source.url} target="_blank" rel="noopener noreferrer">
            {source.title}
            <Icon name="arrow" size={12} />
          </a>
        ) : (
          source.title
        )}
      </h3>
      {source.authors && source.authors.length > 0 && (
        <p className="source-authors">{source.authors.join(", ")}</p>
      )}
      {source.fulltext_status && (
        <small className="source-status">
          {source.fulltext_status === "ready"
            ? "전문 수집 완료"
            : "전문 미확보"}
        </small>
      )}
      {excerpt && source.excerpt && (
        <blockquote className="source-excerpt">{source.excerpt}</blockquote>
      )}
      {excerpt && source.sentence_ids?.length ? (
        <small className="source-location">
          문장 {source.sentence_ids.join(", ")}
        </small>
      ) : null}
    </article>
  );
}
