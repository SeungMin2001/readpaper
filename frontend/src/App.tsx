import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type DragEvent } from 'react'
import * as pdfjs from 'pdfjs-dist'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { api } from './api'
import CitationGraph from './CitationGraph'
import PdfReader from './PdfReader'
import type { ActivityEvent, Answer, GlossPreparation, Paper, Source, VocabularyItem, Word, WordGloss } from './types'

pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.mjs', import.meta.url).toString()

type Panel = 'assistant' | 'research' | 'vocabulary'
type ChatTurn = { id: number; question: string; answer?: Answer; pending?: boolean }

const sourceLabels: Record<Source['kind'], string> = {
  paper: '현재 논문', referenced: '참고 논문', related: '관련 연구',
  citing: '인용한 논문', recent: '최신 연구', background: '배경 자료',
}

function Icon({ name, size = 18 }: { name: 'upload' | 'book' | 'chat' | 'network' | 'words' | 'send' | 'spark' | 'check' | 'search' | 'arrow'; size?: number }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true as const }
  switch (name) {
    case 'upload': return <svg {...common}><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/><path d="M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5"/></svg>
    case 'book': return <svg {...common}><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 0 4 22V5.5Z"/><path d="M4 17a2.5 2.5 0 0 1 2.5-2.5H20M8 7h8"/></svg>
    case 'chat': return <svg {...common}><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H5l-1 2v-6.5A7.5 7.5 0 1 1 20 11.5Z"/><path d="M8 11h8M8 14h5"/></svg>
    case 'network': return <svg {...common}><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="18" r="2.5"/><circle cx="19" cy="18" r="2.5"/><path d="m10.8 7.2-4.5 8.6m6.9-8.6 4.5 8.6M7.5 18h9"/></svg>
    case 'words': return <svg {...common}><path d="M4 6h16M4 12h10M4 18h7"/><path d="m17 15 3 6m-6 0 3-6m-2.1 4h4.2"/></svg>
    case 'send': return <svg {...common}><path d="m21 3-7.2 18-3.6-7.2L3 10.2 21 3Z"/><path d="M10.2 13.8 15 9"/></svg>
    case 'spark': return <svg {...common}><path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3Z"/><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z"/></svg>
    case 'check': return <svg {...common}><path d="m5 12 4 4L19 6"/></svg>
    case 'search': return <svg {...common}><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5"/></svg>
    case 'arrow': return <svg {...common}><path d="M5 12h14m-6-6 6 6-6 6"/></svg>
  }
}

function Brand() {
  return <div className="brand"><span className="brand-mark">r</span><span>readpaper</span></div>
}

export default function App() {
  const [user, setUser] = useState<string | null | undefined>(undefined)
  const [loginName, setLoginName] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [paperId, setPaperId] = useState<string | null>(null)
  const [paper, setPaper] = useState<Paper | null>(null)
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null)
  const [fileName, setFileName] = useState('')
  const [pageIndex, setPageIndex] = useState(0)
  const [glossPreparation, setGlossPreparation] = useState<GlossPreparation | null>(null)
  const [skipPreparation, setSkipPreparation] = useState(false)
  const [preparationAttempt, setPreparationAttempt] = useState(0)
  const [panel, setPanel] = useState<Panel>('assistant')
  const [gloss, setGloss] = useState<{ original: string; meaning: string; sentence: string; contextCount?: number; elapsedMs?: number } | null>(null)
  const [translated, setTranslated] = useState<Set<string>>(new Set())
  const [pendingWordIds, setPendingWordIds] = useState<Set<string>>(new Set())
  const [vocabulary, setVocabulary] = useState<VocabularyItem[]>([])
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [asking, setAsking] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [notice, setNotice] = useState('')
  const [manualTerm, setManualTerm] = useState('')
  const [manualMeaning, setManualMeaning] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)
  const chatScroll = useRef<HTMLDivElement>(null)
  const wordRequests = useRef(new Map<string, Promise<WordGloss>>())

  useEffect(() => {
    api.me().then((result) => setUser(result.username)).catch(() => setUser(null))
  }, [])

  const refreshVocabulary = useCallback(() => {
    void api.vocabulary().then(setVocabulary).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (user) refreshVocabulary()
  }, [user, refreshVocabulary])

  useEffect(() => {
    if (!paperId || paper?.status === 'ready' || paper?.status === 'failed') return
    let active = true
    const poll = () => {
      void api.paper(paperId).then((result) => {
        if (active) setPaper((previous) => previous?.id === result.id ? {
          ...result,
          word_glosses: { ...previous.word_glosses, ...result.word_glosses },
        } : result)
      }).catch((error: Error) => {
        if (active) setNotice(error.message)
      })
    }
    poll()
    const timer = window.setInterval(poll, 2000)
    return () => { active = false; window.clearInterval(timer) }
  }, [paperId, paper?.status])

  useEffect(() => {
    if (!paperId) return
    let active = true
    let timer: number | undefined
    const poll = () => {
      void api.glossPreparation(paperId).then((result) => {
        if (!active) return
        setGlossPreparation(result)
        setPaper((previous) => previous?.id === paperId ? {
          ...previous,
          word_glosses: { ...previous.word_glosses, ...result.word_glosses },
        } : previous)
        if (result.status === 'completed' || result.status === 'failed') window.clearInterval(timer)
      }).catch((error: Error) => {
        if (!active) return
        window.clearInterval(timer)
        setGlossPreparation({ status: 'failed', total: 0, ready: 0, failed: 0,
          error: error.message, word_glosses: {} })
      })
    }
    poll()
    timer = window.setInterval(poll, 1500)
    return () => { active = false; window.clearInterval(timer) }
  }, [paperId, preparationAttempt])

  useEffect(() => {
    chatScroll.current?.scrollTo({ top: chatScroll.current.scrollHeight, behavior: 'smooth' })
  }, [turns, asking])

  const sentenceMap = useMemo(
    () => new Map((paper?.sentences || []).map((sentence) => [sentence.id, sentence])),
    [paper?.sentences],
  )
  const currentPage = paper?.pages?.[pageIndex]
  const canAsk = Boolean(paperId && paper?.status === 'ready')

  async function doLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setNotice('')
    try {
      const result = await api.login(loginName, loginPassword)
      setUser(result.username)
      setLoginPassword('')
    } catch (error) { setNotice((error as Error).message) }
  }

  async function upload(file: File) {
    setNotice('')
    setUploading(true)
    setPdf(null)
    setFileName('')
    setPaper(null)
    setPaperId(null)
    setTranslated(new Set())
    setPendingWordIds(new Set())
    wordRequests.current.clear()
    setGloss(null)
    setTurns([])
    setPanel('assistant')
    setPageIndex(0)
    setGlossPreparation(null)
    setSkipPreparation(false)
    try {
      if (file.size > 25 * 1024 * 1024) throw new Error('PDF는 25MB 이하로 업로드해 주세요.')
      const bytes = new Uint8Array(await file.arrayBuffer())
      const loaded = await pdfjs.getDocument({ data: bytes }).promise
      setPdf(loaded)
      setFileName(file.name)
      const result = await api.upload(file)
      setPaperId(result.paper_id)
      setPaper(await api.paper(result.paper_id))
      if (result.cached) setNotice('이전에 분석한 논문을 불러왔습니다.')
    } catch (error) {
      setPdf(null)
      setPaper(null)
      setPaperId(null)
      setNotice((error as Error).message)
    }
    finally { setUploading(false) }
  }

  async function retryPreparation() {
    if (!paperId) return
    try {
      const result = await api.retryGlossPreparation(paperId)
      setGlossPreparation(result)
      setPreparationAttempt((previous) => previous + 1)
    } catch (error) { setNotice((error as Error).message) }
  }

  function revealGloss(unit: WordGloss, word: Word, contextCount?: number, elapsedMs?: number) {
    const key = unit.word_ids.join(':')
    const wasTranslated = translated.has(key)
    setTranslated((previous) => {
      const next = new Set(previous)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
    if (wasTranslated) {
      setGloss(null)
      return
    }
    const sentence = sentenceMap.get(unit.sentence_id || word.sentence_id)
    setGloss({ original: unit.original, meaning: unit.meaning, sentence: sentence?.text || '', contextCount, elapsedMs })
    setPanel('assistant')
    if (paper) {
      void api.saveWord({
        paper_title: paper.title, term: unit.original,
        meaning: unit.meaning, source_sentence: sentence?.text || '',
      }).then(refreshVocabulary).catch((error: Error) => setNotice(`단어 저장 실패: ${error.message}`))
    }
  }

  function toggleWord(word: Word, cached?: WordGloss) {
    if (cached) {
      revealGloss(cached, word)
      return
    }
    if (!paperId || wordRequests.current.has(word.id)) return

    setPendingWordIds((previous) => new Set(previous).add(word.id))
    const request = api.translateWord(paperId, word.id).then((result) => {
      const unit: WordGloss = {
        word_ids: result.word_ids, original: result.original,
        meaning: result.meaning, sentence_id: result.sentence_id,
      }
      setPaper((previous) => previous ? {
        ...previous,
        activity: result.activity || previous.activity,
        word_glosses: {
          ...previous.word_glosses,
          ...Object.fromEntries(unit.word_ids.map((id) => [id, unit])),
        },
      } : previous)
      revealGloss(unit, word, result.context_count, result.elapsed_ms)
      return unit
    }).catch((error: Error) => {
      setNotice(error.message)
      throw error
    }).finally(() => {
      wordRequests.current.delete(word.id)
      setPendingWordIds((previous) => {
        const next = new Set(previous)
        next.delete(word.id)
        return next
      })
    })
    wordRequests.current.set(word.id, request)
    void request.catch(() => undefined)
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const prompt = question.trim()
    if (!paperId || !prompt || !canAsk) return
    const id = Date.now()
    setTurns((previous) => [...previous, { id, question: prompt, pending: true }])
    setQuestion('')
    setAsking(true)
    setNotice('')
    try {
      const result = await api.ask(paperId, prompt)
      setTurns((previous) => previous.map((turn) => turn.id === id ? { ...turn, answer: result, pending: false } : turn))
    } catch (error) {
      setTurns((previous) => previous.map((turn) => turn.id === id ? { ...turn, pending: false } : turn))
      setNotice((error as Error).message)
    } finally { setAsking(false) }
  }

  async function saveManual(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    try {
      await api.saveWord({
        paper_title: paper?.title || '직접 추가', term: manualTerm,
        meaning: manualMeaning, source_sentence: '',
      })
      setManualTerm('')
      setManualMeaning('')
      refreshVocabulary()
    } catch (error) { setNotice((error as Error).message) }
  }

  function handleDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) void upload(file)
  }

  if (user === undefined) return <div className="boot-screen"><span className="boot-mark">r</span><span>readpaper</span></div>

  if (!user) return (
    <div className="login-page">
      <div className="login-card">
        <Brand />
        <p className="eyebrow">A QUIETER WAY TO READ</p>
        <h1>논문을 읽는<br /><em>나만의 자리</em></h1>
        <p className="muted">읽다가 막히는 순간, 맥락을 잃지 않고 뜻을 확인하세요.</p>
        <form onSubmit={doLogin} className="login-form">
          <label>아이디<input value={loginName} onChange={(event) => setLoginName(event.target.value)} autoComplete="username" required /></label>
          <label>비밀번호<input type="password" value={loginPassword} onChange={(event) => setLoginPassword(event.target.value)} autoComplete="current-password" required /></label>
          <button className="primary-button" type="submit">읽기 시작하기 <Icon name="arrow" size={16} /></button>
        </form>
        {notice && <p className="inline-error">{notice}</p>}
      </div>
      <div className="login-side"><span className="login-index">READPAPER / 01</span><p>Good research<br />takes <em>time.</em></p><span className="login-caption">TAKE YOURS.</span></div>
    </div>
  )

  const sourceCounts = countSources(paper?.sources || [])
  const preparingGlosses = Boolean(pdf && !skipPreparation &&
    (uploading || !paper || glossPreparation?.status !== 'completed'))

  return (
    <div className="app-shell">
      <header className="site-header">
        <Brand />
        <div className="header-right">
          {paper?.status === 'ready' && !preparingGlosses && <span className="header-state"><span className="status-dot" /> 읽을 준비가 됐어요</span>}
          <span className="user-name">{user}</span>
          <button className="text-button" onClick={() => { void api.logout().then(() => { setUser(null); setPaper(null); setPaperId(null); setPdf(null); setFileName('') }) }}>로그아웃</button>
        </div>
      </header>

      {!pdf ? (
        <main className="welcome-page">
          <section className="welcome-copy">
            <p className="eyebrow"><span className="eyebrow-line" /> A SPACE FOR SLOW READING</p>
            <h1>논문을 읽는 일에<br />더 많은 <em>여백</em>을.</h1>
            <p className="welcome-description">원문을 따라가다 모르는 단어를 만나면 바로 뜻을 보고,<br className="desktop-break" /> 이해가 막히면 논문에 다시 물어보세요.</p>
            <div className="welcome-notes"><span><i>01</i> 원문 그대로 읽기</span><span><i>02</i> 문맥 안에서 뜻 확인</span><span><i>03</i> 인용 관계 따라가기</span></div>
          </section>
          <section
            className={`upload-card ${dragging ? 'upload-card--dragging' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div className="upload-card-top"><span>START WITH A PAPER</span><span>PDF / UP TO 25 MB</span></div>
            <div className="upload-illustration" aria-hidden="true"><div className="upload-sheet"><span /><span /><span /><span /></div><div className="upload-arrow"><Icon name="upload" size={21} /></div></div>
            <h2>읽고 싶은 논문을<br />여기에 놓아주세요.</h2>
            <p>텍스트를 선택할 수 있는 PDF 파일을 업로드합니다.</p>
            <button className="primary-button" disabled={uploading} onClick={() => fileInput.current?.click()}>{uploading ? 'PDF를 여는 중…' : 'PDF 파일 선택'}<Icon name="arrow" size={16} /></button>
            <div className="drop-hint">또는 파일을 이 영역으로 끌어오세요</div>
          </section>
          <input ref={fileInput} hidden type="file" accept="application/pdf,.pdf" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.currentTarget.value = '' }} />
          <div className="welcome-footer"><span>READPAPER — RESEARCH, AT YOUR PACE</span><span>01 / YOUR READING SPACE</span></div>
          {notice && <div className="toast" role="status">{notice}<button onClick={() => setNotice('')} aria-label="닫기">×</button></div>}
        </main>
      ) : preparingGlosses ? (
        <main className="preparation-page" aria-live="polite">
          <div className="preparation-card">
            <p className="eyebrow">PREPARING YOUR PAPER</p>
            <h1>{glossPreparation?.status === 'failed' ? '용어 준비가 멈췄습니다.' : '읽기 전에 용어를 준비하고 있어요.'}</h1>
            <p className="preparation-title">{paper?.title || fileName}</p>
            <div className="preparation-progress" role="progressbar" aria-label="주요 용어 준비" aria-valuemin={0}
              aria-valuemax={glossPreparation?.total || 1} aria-valuenow={glossPreparation?.ready || 0}>
              <span style={{ width: `${glossPreparation?.total ? glossPreparation.ready / glossPreparation.total * 100 : 0}%` }} />
            </div>
            <p className="preparation-count"><strong>{glossPreparation?.ready || 0}</strong> / {glossPreparation?.total ?? '…'}개 주요 용어 준비됨</p>
            <p className="preparation-note">논문 전체에서 주요 용어를 고른 뒤 문맥에 맞는 뜻을 저장합니다. 나머지 단어는 읽다가 누르면 번역합니다.</p>
            {glossPreparation?.status === 'failed' && <div className="preparation-actions">
              <p>{glossPreparation.error || '로컬 모델 연결을 확인해 주세요.'}</p>
              <button className="primary-button" onClick={() => { void retryPreparation() }}>다시 시도</button>
              <button className="secondary-button" onClick={() => setSkipPreparation(true)}>원문 먼저 읽기</button>
            </div>}
          </div>
        </main>
      ) : (
        <main className="workspace">
          <div className="paper-toolbar">
            <div className="paper-heading-icon"><Icon name="book" size={20} /></div>
            <div className="paper-title-block"><p className="eyebrow">NOW READING</p><h1 title={paper?.title}>{paper?.title || fileName}</h1><span>{fileName}</span></div>
            <div className="toolbar-actions">
              {paper?.status === 'ready' && <span className="ready-label"><span className="status-dot" /> READY</span>}
              <button className="secondary-button" onClick={() => fileInput.current?.click()}><span>＋</span> 새 논문</button>
            </div>
            <input ref={fileInput} hidden type="file" accept="application/pdf,.pdf" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.currentTarget.value = '' }} />
          </div>

          <AnalysisTimeline paper={paper} paperId={paperId} uploading={uploading} onRetry={() => {
            if (!paperId || !paper) return
            void api.retry(paperId).then(() => setPaper({ ...paper, status: 'processing', progress: 0, error: null })).catch((error: Error) => setNotice(error.message))
          }} />

          <div className="reader-layout">
            <section className="reader-main" aria-label="논문 읽기">
              <div className="reader-controls">
                <div className="reader-label"><span className="live-dot" /> ORIGINAL PAPER</div>
                <div className="page-nav">
                  <button disabled={pageIndex === 0} onClick={() => setPageIndex(Math.max(0, pageIndex - 1))} aria-label="이전 페이지">‹</button>
                  <span>{pageIndex + 1}<i>/</i>{pdf.numPages}</span>
                  <button disabled={pageIndex + 1 >= pdf.numPages} onClick={() => setPageIndex(Math.min(pdf.numPages - 1, pageIndex + 1))} aria-label="다음 페이지">›</button>
                </div>
              </div>
              <p className="reader-hint">준비된 주요 용어는 누르면 바로 뜻이 보입니다. 다른 단어는 논문 문맥을 반영해 번역합니다.{glossPreparation?.failed ? <span> · {glossPreparation.failed}개 용어는 미리 준비되지 않았습니다.</span> : null}{paper?.status !== 'ready' && <span> · 문헌 검색은 계속 진행 중입니다.</span>}</p>
              <div className="pdf-stage">
                <PdfReader
                  document={pdf} pageIndex={pageIndex} page={currentPage}
                  wordGlosses={{ ...paper?.word_glosses, ...glossPreparation?.word_glosses }}
                  translated={translated} pendingWordIds={pendingWordIds}
                  onWord={toggleWord}
                />
              </div>
              <div className="reader-foot"><span><Icon name="book" size={14} /> {paper?.pages?.length || pdf.numPages} PAGES</span><span>클릭한 단어는 내 단어장에 저장됩니다</span></div>
            </section>

            <aside className="research-dock" aria-label="논문 도구">
              <div className="dock-header">
                <div><p className="eyebrow">READING DESK</p><h2>논문과 함께</h2></div>
                <span className="dock-status"><i /> LOCAL</span>
              </div>
              <div className="dock-tabs" role="tablist" aria-label="읽기 도구">
                <button role="tab" aria-selected={panel === 'assistant'} className={panel === 'assistant' ? 'active' : ''} onClick={() => setPanel('assistant')}><Icon name="chat" size={16} /><span>대화</span></button>
                <button role="tab" aria-selected={panel === 'research'} className={panel === 'research' ? 'active' : ''} onClick={() => setPanel('research')}><Icon name="network" size={16} /><span>인용 지도</span>{sourceCounts.references + sourceCounts.citing > 0 && <b>{sourceCounts.references + sourceCounts.citing}</b>}</button>
                <button role="tab" aria-selected={panel === 'vocabulary'} className={panel === 'vocabulary' ? 'active' : ''} onClick={() => setPanel('vocabulary')}><Icon name="words" size={16} /><span>단어장</span></button>
              </div>

              {panel === 'assistant' && <section className="dock-panel assistant-panel" role="tabpanel">
                <div className="assistant-intro"><span className="assistant-avatar"><Icon name="spark" size={17} /></span><div><strong>논문 읽기 도우미</strong><p>논문의 내용과 수집한 자료를 바탕으로 답해요.</p></div></div>
                {gloss && <div className="gloss-card"><div className="gloss-heading"><span>WORD IN CONTEXT</span><button onClick={() => setGloss(null)} aria-label="뜻 패널 닫기">×</button></div><div className="gloss-term">{gloss.original}<span>{gloss.meaning}</span></div>{gloss.sentence && <p>{gloss.sentence}</p>}<small className="gloss-context-note">논문 문맥 {gloss.contextCount ? `${gloss.contextCount}곳` : '기반'}{gloss.elapsedMs ? ` · ${ (gloss.elapsedMs / 1000).toFixed(1)}초` : ' · 저장된 뜻'}</small></div>}
                <div className="chat-scroll" ref={chatScroll}>
                  {turns.length === 0 ? <div className="chat-empty"><p className="chat-greeting">안녕하세요. 이 논문을 읽다가 막히는 부분이 있나요?</p><p>방법, 용어, 주장 사이의 관계를 물어보세요. 답변에는 근거 자료를 함께 표시합니다.</p><div className="prompt-list">{[
                    '이 논문이 해결하려는 문제는 무엇인가요?',
                    '핵심 방법을 간단히 설명해줘',
                    '이 연구의 한계는 무엇인가요?',
                  ].map((prompt) => <button key={prompt} onClick={() => setQuestion(prompt)} disabled={!canAsk}>{prompt}<Icon name="arrow" size={13} /></button>)}</div></div> : <div className="chat-turns">{turns.map((turn) => <div className="chat-turn" key={turn.id}>
                    <div className="user-bubble">{turn.question}</div>
                    {turn.pending ? <div className="thinking-line"><span /><span /><span /> 답변을 준비하고 있어요</div> : turn.answer ? <div className="assistant-bubble"><p>{turn.answer.answer}</p>{turn.answer.sources.length > 0 && <div className="answer-sources"><span>근거 자료</span>{turn.answer.sources.map((source) => <SourceCard key={source.id} source={source} compact excerpt />)}</div>}</div> : null}
                  </div>)}</div>}
                </div>
                <form className="chat-form" onSubmit={ask}>
                  {!canAsk && <p className="chat-disabled-note">논문 분석이 끝나면 질문할 수 있어요.</p>}
                  <div className="chat-input-wrap"><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={canAsk ? '논문에 대해 질문하기…' : '분석이 끝나면 질문을 입력하세요'} disabled={!canAsk || asking} /><button type="submit" disabled={!canAsk || asking || !question.trim()} aria-label="질문 보내기"><Icon name="send" size={17} /></button></div>
                  <div className="chat-disclaimer">답변은 논문 자료를 검색해 구성됩니다. 원문과 근거를 함께 확인하세요.</div>
                </form>
              </section>}

              {panel === 'research' && <ResearchPanel title={paper?.title || fileName} sources={paper?.sources || []} warnings={paper?.source_warnings || []} status={paper?.status || 'processing'} sourcesReady={Boolean(paper?.sources_ready)} error={paper?.error || undefined} />}
              {panel === 'vocabulary' && <div className="dock-panel vocabulary-panel" role="tabpanel"><div className="panel-title-row"><div><p className="eyebrow">YOUR READING NOTES</p><h2>나의 단어장 <span className="count">{vocabulary.length}</span></h2></div></div><p className="panel-copy">논문을 읽으며 확인한 단어가 모입니다.</p><form className="manual-form" onSubmit={saveManual}><input value={manualTerm} onChange={(event) => setManualTerm(event.target.value)} placeholder="단어" required /><input value={manualMeaning} onChange={(event) => setManualMeaning(event.target.value)} placeholder="뜻" required /><button className="secondary-button" type="submit">추가</button></form><div className="vocab-list">{vocabulary.map((item) => <article className="vocab-item" key={item.id}><div><strong>{item.term}</strong><span>{item.meaning}</span></div>{item.source_sentence && <p>{item.source_sentence}</p>}<small>{item.paper_title}</small><button aria-label={`${item.term} 삭제`} onClick={() => { void api.deleteWord(item.id).then(refreshVocabulary).catch((error: Error) => setNotice(error.message)) }}>×</button></article>)}{vocabulary.length === 0 && <div className="empty-research"><span><Icon name="words" size={24} /></span><p>논문에서 모르는 단어를 눌러보세요.</p></div>}</div></div>}
            </aside>
          </div>
          {notice && <div className="toast" role="status">{notice}<button onClick={() => setNotice('')} aria-label="닫기">×</button></div>}
        </main>
      )}
    </div>
  )
}

function countSources(sources: Source[]) {
  return {
    references: sources.filter((source) => source.kind === 'referenced').length,
    citing: sources.filter((source) => source.kind === 'citing').length,
    related: sources.filter((source) => source.kind === 'related' || source.kind === 'recent').length,
  }
}

function AnalysisTimeline({ paper, paperId, uploading, onRetry }: { paper: Paper | null; paperId: string | null; uploading: boolean; onRetry: () => void }) {
  const status = paper?.status
  const activity = paper?.activity || []
  const sourcesReady = Boolean(paper?.sources_ready)
  const completedSteps = status === 'ready' ? 3 : sourcesReady ? 2 : paperId ? 1 : 0
  const steps = ['논문 구조 읽기', '인용·관련 연구 찾기', '논문 문맥 검색 준비']
  const counts = countSources(paper?.sources || [])
  return <section className={`analysis-timeline ${status === 'failed' ? 'analysis-timeline--failed' : ''}`} aria-label="논문 분석 진행">
    <div className="timeline-summary"><div className="timeline-title"><span className="timeline-symbol"><Icon name={status === 'ready' ? 'check' : 'network'} size={17} /></span><div><span className="timeline-kicker">PAPER PROCESS</span><strong>{status === 'ready' ? '논문을 읽을 수 있습니다' : status === 'failed' ? sourcesReady ? '문헌 수집 완료 · 문맥 준비 오류' : '문헌 분석을 마치지 못했습니다' : sourcesReady ? '문헌을 찾았습니다. 검색 문맥을 준비하고 있어요' : uploading && !paperId ? 'PDF를 읽을 준비 중입니다' : '인용·관련 연구를 찾고 있어요'}</strong></div></div>
      <div className="timeline-detail">{sourcesReady ? <><b>{counts.references}</b>편 참고 <i>·</i> <b>{counts.citing}</b>편 피인용 <i>·</i> <b>{counts.related}</b>편 관련 자료{status === 'failed' && paper?.error ? <span> · {paper.error}</span> : null}</> : status === 'failed' ? paper?.error : '원문 문맥과 인용·관련 연구를 분석하고 있습니다.'}</div>
    </div>
    <div className="timeline-steps">{steps.map((label, index) => {
      const complete = index < completedSteps
      const failed = status === 'failed' && index === completedSteps
      const active = status !== 'failed' && index === completedSteps
      return <div className={`timeline-step ${complete ? 'complete' : ''} ${active ? 'active' : ''} ${failed ? 'failed' : ''}`} key={label}><span className="timeline-step-marker">{complete ? <Icon name="check" size={13} /> : failed ? '!' : String(index + 1).padStart(2, '0')}</span><div className="timeline-step-copy"><strong>{label}</strong><span>{index === 0 ? '제목과 문장 구조' : index === 1 ? 'OpenAlex 관계 정보' : '원문 문맥을 반영'}</span></div></div>
    })}</div>
    {(paperId || uploading) && <ActivityLog activity={activity} live={uploading || status === 'processing'} />}
    {status === 'failed' && paperId && <button className="retry-button" onClick={onRetry}>다시 시도</button>}
  </section>
}

function ActivityLog({ activity, live }: { activity: ActivityEvent[]; live: boolean }) {
  const entries = [...activity].reverse()
  return <section className="timeline-activity" aria-label="실시간 분석 로그">
    <header className="timeline-activity-heading"><div><span className="eyebrow">LIVE PROCESS LOG</span><strong>수집·분석 로그</strong></div><span className={`activity-live ${live ? 'activity-live--on' : ''}`}><i />{live ? 'LIVE' : '기록'}</span></header>
    <div className="timeline-activity-list" role="log" aria-live="polite" aria-relevant="additions">
      {entries.length > 0 ? entries.map((event) => <div className={`activity-entry activity-entry--${event.state}`} key={event.id}>
        <time dateTime={new Date(event.created_at * 1000).toISOString()}>{new Date(event.created_at * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}</time>
        <i aria-hidden="true" />
        <span>{event.message}</span>
      </div>) : <p className="activity-empty">PDF를 읽고 있습니다. 첫 처리 로그를 기다리는 중입니다.</p>}
    </div>
  </section>
}

function ResearchPanel({ title, sources, warnings, status, sourcesReady, error }: { title: string; sources: Source[]; warnings: string[]; status: Paper['status']; sourcesReady: boolean; error?: string }) {
  const [query, setQuery] = useState('')
  const counts = countSources(sources)
  const citationsReady = sourcesReady || status === 'ready'
  const related = sources.filter((source) => source.kind === 'related' || source.kind === 'recent')
  const searchable = sources.filter((source) => source.kind !== 'background')
  const filtered = searchable.filter((source) => `${source.title} ${(source.authors || []).join(' ')}`.toLowerCase().includes(query.toLowerCase()))
  return <div className="dock-panel research-panel" role="tabpanel">
    <div className="research-panel-heading"><p className="eyebrow">PAPER CONNECTIONS</p><h2>인용 지도</h2><p>논문 사이의 인용 관계를 한눈에 봅니다.</p></div>
    {!citationsReady ? <div className={`graph-wait ${status === 'failed' ? 'graph-wait--failed' : ''}`}><div className="graph-wait-mark"><Icon name="network" size={26} /></div><strong>{status === 'failed' ? '문헌 관계를 가져오지 못했습니다' : '관계를 찾고 있습니다'}</strong><p>{status === 'failed' ? error || '검색을 완료하지 못했습니다. 다시 시도해 주세요.' : '분석이 끝나면 참고문헌과 피인용 논문을 그래프로 보여드릴게요.'}</p></div> : <>
      {status === 'processing' && <p className="collection-ready-note">문헌 검색이 끝나면 논문 문맥 검색을 준비합니다.</p>}
      {status === 'failed' && error && <p className="source-warning">문헌 정보는 보존했습니다. 이후 분석 단계에서 오류가 발생했습니다: {error}</p>}
      <div className="graph-counts"><div><strong>{counts.references}</strong><span>찾은 참고문헌</span></div><div><strong>{counts.citing}</strong><span>찾은 피인용</span></div><div><strong>{counts.related}</strong><span>관련 연구 후보</span></div></div>
      <div className="citation-graph-card"><div className="graph-card-heading"><div><strong>직접 인용 관계</strong><span>화살표는 인용 방향을 나타냅니다</span></div><div className="graph-legend"><i className="legend-reference" />인용<i className="legend-citing" />피인용</div></div><CitationGraph title={title} sources={sources} /></div>
      <div className="research-results-heading"><div><p className="eyebrow">DISCOVERED PAPERS</p><h3>찾은 연구 <span>{searchable.length}</span></h3></div><label className="source-search"><Icon name="search" size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="제목으로 검색" /></label></div>
      {warnings.map((warning) => <p className="source-warning" key={warning}>{warning}</p>)}
      {related.length > 0 && <div className="related-note"><strong>관련 연구 후보</strong><span>OpenAlex의 관련도 순서에서 최대 8편을 표시합니다. 인용 기록이 확인된 연결만 그래프에 표시합니다.</span></div>}
      <div className="source-list">{filtered.map((source) => <SourceCard key={source.id} source={source} compact />)}{filtered.length === 0 && <p className="empty-note">검색 결과가 없습니다.</p>}</div>
      <p className="source-footnote">자료마다 초록 또는 서지 정보만 제공될 수 있습니다. 전문 수집 여부는 별도로 확인하세요.</p>
    </>}
  </div>
}

function SourceCard({ source, excerpt = false, compact = false }: { source: Source; excerpt?: boolean; compact?: boolean }) {
  const scope = source.content_scope === 'full_text' ? '전문' : source.content_scope === 'abstract' ? '초록' : source.content_scope === 'metadata' ? '서지 정보' : null
  return <article className={`source-card ${compact ? 'source-card--compact' : ''}`}><div className="source-card-top"><span className={`source-kind source-kind--${source.kind}`}>{sourceLabels[source.kind]}</span>{scope && <span>{scope}</span>}{source.year && <span>{source.year}</span>}</div><h3>{source.url ? <a href={source.url} target="_blank" rel="noopener noreferrer">{source.title}<Icon name="arrow" size={12} /></a> : source.title}</h3>{source.authors && source.authors.length > 0 && <p className="source-authors">{source.authors.join(', ')}</p>}{excerpt && source.excerpt && <p className="source-excerpt">{source.excerpt}</p>}</article>
}
