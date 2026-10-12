import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as pdfjs from 'pdfjs-dist'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { api } from './api'
import PdfReader from './PdfReader'
import type { Answer, Paper, Sentence, Source, TranslationUnit, VocabularyItem, Word } from './types'

pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.mjs', import.meta.url).toString()

type Tab = 'translation' | 'research' | 'questions' | 'vocabulary'

const sourceLabels: Record<Source['kind'], string> = {
  paper: '현재 논문', referenced: '참고 논문', related: '관련 논문',
  citing: '이 논문을 인용', recent: '최신 관련 논문', background: '배경 지식',
}

function Brand() {
  return <div className="brand"><span className="brand-mark">r<span>.</span></span><span>readpaper</span></div>
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
  const [tab, setTab] = useState<Tab>('translation')
  const [mode, setMode] = useState<'word' | 'sentence'>('word')
  const [selectedSentence, setSelectedSentence] = useState<string | null>(null)
  const [translated, setTranslated] = useState<Set<string>>(new Set())
  const [vocabulary, setVocabulary] = useState<VocabularyItem[]>([])
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [asking, setAsking] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [notice, setNotice] = useState('')
  const [manualTerm, setManualTerm] = useState('')
  const [manualMeaning, setManualMeaning] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

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
        if (active) setPaper(result)
      }).catch((error: Error) => {
        if (active) setNotice(error.message)
      })
    }
    poll()
    const timer = window.setInterval(poll, 2500)
    return () => { active = false; window.clearInterval(timer) }
  }, [paperId, paper?.status])

  useEffect(() => {
    if (!paperId || paper?.status !== 'ready') return
    const timer = window.setInterval(() => {
      void api.paper(paperId).catch(() => undefined)
    }, 5 * 60 * 1000)
    return () => window.clearInterval(timer)
  }, [paperId, paper?.status])

  const sentenceMap = useMemo(
    () => new Map((paper?.sentences || []).map((sentence) => [sentence.id, sentence])),
    [paper?.sentences],
  )
  const currentSentence = selectedSentence ? sentenceMap.get(selectedSentence) : undefined
  const currentPage = paper?.pages?.[pageIndex]

  async function doLogin(event: React.FormEvent<HTMLFormElement>) {
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
    setPaper(null)
    setPaperId(null)
    setPdf(null)
    setTranslated(new Set())
    setSelectedSentence(null)
    setAnswer(null)
    try {
      const bytes = new Uint8Array(await file.arrayBuffer())
      const loaded = await pdfjs.getDocument({ data: bytes }).promise
      const result = await api.upload(file)
      setPdf(loaded)
      setFileName(file.name)
      setPageIndex(0)
      setPaperId(result.paper_id)
      setPaper(await api.paper(result.paper_id))
      if (result.cached) setNotice('기존 분석 결과를 불러왔습니다.')
    } catch (error) { setNotice((error as Error).message) }
    finally { setUploading(false) }
  }

  function chooseSentence(id: string) {
    setSelectedSentence(id)
    setTab('translation')
  }

  function toggleWord(key: string, unit: TranslationUnit, word: Word) {
    const wasTranslated = translated.has(key)
    setTranslated((previous) => {
      const next = new Set(previous)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
    if (!wasTranslated && paper) {
      const sentence = sentenceMap.get(word.sentence_id)
      void api.saveWord({
        paper_title: paper.title, term: unit.original,
        meaning: unit.meaning, source_sentence: sentence?.text || '',
      }).then(refreshVocabulary).catch((error: Error) => setNotice(`단어 저장 실패: ${error.message}`))
    }
  }

  async function ask(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!paperId || !question.trim()) return
    setAsking(true)
    setAnswer(null)
    setNotice('')
    try { setAnswer(await api.ask(paperId, question.trim())) }
    catch (error) { setNotice((error as Error).message) }
    finally { setAsking(false) }
  }

  async function saveManual(event: React.FormEvent<HTMLFormElement>) {
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

  if (user === undefined) return <div className="center-message">readpaper를 준비하고 있습니다…</div>

  if (!user) return (
    <div className="login-page">
      <div className="login-card">
        <Brand />
        <p className="eyebrow">YOUR RESEARCH READING SPACE</p>
        <h1>영문 논문을<br />내 속도로 읽다.</h1>
        <p className="muted">논문의 맥락을 이해한 번역으로, 모르는 단어에서 멈추지 마세요.</p>
        <form onSubmit={doLogin} className="login-form">
          <label>아이디<input value={loginName} onChange={(event) => setLoginName(event.target.value)} autoComplete="username" required /></label>
          <label>비밀번호<input type="password" value={loginPassword} onChange={(event) => setLoginPassword(event.target.value)} autoComplete="current-password" required /></label>
          <button className="primary-button" type="submit">읽기 시작하기 <span>↗</span></button>
        </form>
        {notice && <p className="inline-error">{notice}</p>}
      </div>
      <div className="login-art" aria-hidden="true"><span className="art-circle" /><span className="art-quote">Read between<br />the lines.</span><span className="art-small">RESEARCH, REIMAGINED.</span></div>
    </div>
  )

  return (
    <div className="app-shell">
      <header className="site-header">
        <Brand />
        <div className="header-right">
          {paper?.status === 'ready' && <span className="status-pill"><span className="status-dot" /> 분석 완료</span>}
          <span className="user-name">{user}</span>
          <button className="text-button" onClick={() => { void api.logout().then(() => { setUser(null); setPaper(null); setPdf(null) }) }}>로그아웃</button>
        </div>
      </header>

      {!paperId ? (
        <main className="welcome">
          <div className="welcome-copy">
            <p className="eyebrow"><span className="eyebrow-line" /> YOUR PAPER, YOUR PACE</p>
            <h1>영문 논문을<br /><em>내 속도로</em> 읽다.</h1>
            <p>원문의 흐름은 유지하고, 필요한 순간에만 뜻을 확인합니다.<br />논문을 이해한 번역과 근거 있는 답변을 한 공간에서.</p>
            <button className="primary-button upload-button" disabled={uploading} onClick={() => fileInput.current?.click()}>
              {uploading ? 'PDF 업로드 중…' : 'PDF 논문 열기'} <span>↗</span>
            </button>
            <p className="upload-note">텍스트를 선택할 수 있는 PDF · 최대 25MB</p>
          </div>
          <div className="welcome-visual" aria-hidden="true">
            <div className="visual-paper">
              <div className="visual-head">ABSTRACT <span>01 / 12</span></div>
              <h2>Understanding complex ideas, one word at a time.</h2>
              <div className="visual-rule" />
              <p>Recent advances in <span className="visual-highlight">representation learning</span> have opened new possibilities for scientific discovery.</p>
              <div className="visual-lines"><i /><i /><i /><i /></div>
              <div className="visual-translation"><span>✓</span> 표현 학습</div>
            </div>
          </div>
          <input ref={fileInput} hidden type="file" accept="application/pdf,.pdf" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.currentTarget.value = '' }} />
          {notice && <p className="inline-error welcome-error">{notice}</p>}
        </main>
      ) : (
        <main className="workspace">
          <div className="paper-toolbar">
            <div className="paper-title-block"><p className="eyebrow">NOW READING</p><h1 title={paper?.title}>{paper?.title || fileName}</h1><span>{fileName}</span></div>
            <button className="secondary-button" onClick={() => fileInput.current?.click()}>+ 새 PDF</button>
            <input ref={fileInput} hidden type="file" accept="application/pdf,.pdf" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.currentTarget.value = '' }} />
          </div>
          {paper?.status !== 'ready' ? (
            <section className="processing-card">
              <div className="processing-orbit">◎</div>
              <p className="eyebrow">PAPER ANALYSIS</p>
              <h2>{paper?.status === 'failed' ? '분석을 마치지 못했습니다' : '논문의 맥락을 읽고 있습니다'}</h2>
              <p>{paper?.status === 'failed' ? paper.error : '문장과 전문 용어를 분석하고, 관련 연구를 수집해 번역을 준비합니다.'}</p>
              {paper?.status !== 'failed' && <div className="progress-track"><div style={{ width: `${paper?.progress || 0}%` }} /></div>}
              {paper?.status !== 'failed' && <span className="progress-label">{paper?.progress || 0}% 완료</span>}
              {paper?.status === 'failed' && <button className="primary-button" onClick={() => { if (paperId) void api.retry(paperId).then(() => setPaper({ ...paper, status: 'processing', progress: 0 })).catch((error: Error) => setNotice(error.message)) }}>다시 시도</button>}
            </section>
          ) : (
            <div className="reader-layout">
              <section className="reader-main">
                <div className="reader-controls">
                  <div className="mode-switch" role="group" aria-label="읽기 모드">
                    <button className={mode === 'word' ? 'active' : ''} onClick={() => setMode('word')}>단어 · 용어</button>
                    <button className={mode === 'sentence' ? 'active' : ''} onClick={() => setMode('sentence')}>문장 선택</button>
                  </div>
                  <div className="page-nav">
                    <button disabled={pageIndex === 0} onClick={() => setPageIndex(pageIndex - 1)} aria-label="이전 페이지">‹</button>
                    <span>{pageIndex + 1} <span className="page-slash">/</span> {paper.pages?.length || pdf?.numPages || 0}</span>
                    <button disabled={pageIndex + 1 >= (paper.pages?.length || 0)} onClick={() => setPageIndex(pageIndex + 1)} aria-label="다음 페이지">›</button>
                  </div>
                </div>
                <p className="reader-hint">{mode === 'word' ? '단어 또는 전문 용어를 눌러 뜻을 확인하세요. 다시 누르면 원문으로 돌아갑니다.' : '문장 안의 단어를 눌러 오른쪽에서 문장 번역을 확인하세요.'}</p>
                {pdf && currentPage && <PdfReader document={pdf} page={currentPage} sentences={paper.sentences || []} translations={paper.translations || {}} translated={translated} selectedSentence={selectedSentence} mode={mode} onWord={toggleWord} onSentence={chooseSentence} />}
              </section>
              <aside className="reader-aside">
                <div className="aside-tabs" role="tablist" aria-label="논문 도구">
                  {([['translation', '문장 번역'], ['research', '관련 자료'], ['questions', '질문'], ['vocabulary', '단어장']] as [Tab, string][]).map(([key, label]) => (
                    <button key={key} role="tab" aria-selected={tab === key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{label}</button>
                  ))}
                </div>
                <div className="aside-content">
                  {tab === 'translation' && <TranslationPanel sentence={currentSentence} translation={currentSentence ? paper.translations?.[currentSentence.id]?.translation : undefined} />}
                  {tab === 'research' && <ResearchPanel sources={paper.sources || []} warnings={paper.source_warnings || []} summary={paper.profile?.summary || ''} />}
                  {tab === 'questions' && <div className="panel-section"><p className="eyebrow">ASK WITH EVIDENCE</p><h2>논문에 질문하기</h2><p className="muted">현재 논문과 수집한 관련 자료를 검색해 근거와 함께 답합니다. 이 기능은 Mac mini의 모델 서버가 켜져 있어야 합니다.</p><form className="question-form" onSubmit={ask}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="예: 이 논문의 방법이 기존 연구와 다른 점은?" rows={4} required /><button className="primary-button" disabled={asking}>{asking ? '근거를 찾는 중…' : '질문하기'} <span>↗</span></button></form>{answer && <div className="answer-box"><p>{answer.answer}</p>{answer.sources.length > 0 && <><h3>근거 자료</h3>{answer.sources.map((source) => <SourceCard key={source.id} source={source} excerpt />)}</>}</div>}</div>}
                  {tab === 'vocabulary' && <div className="panel-section"><p className="eyebrow">YOUR WORDS</p><h2>나의 단어장 <span className="count">{vocabulary.length}</span></h2><p className="muted">뜻을 확인한 단어는 자동으로 모입니다.</p><form className="manual-form" onSubmit={saveManual}><input value={manualTerm} onChange={(event) => setManualTerm(event.target.value)} placeholder="직접 추가할 단어" required /><input value={manualMeaning} onChange={(event) => setManualMeaning(event.target.value)} placeholder="한국어 뜻" required /><button className="secondary-button" type="submit">단어 추가</button></form><div className="vocab-list">{vocabulary.map((item) => <article className="vocab-item" key={item.id}><div><strong>{item.term}</strong><span>{item.meaning}</span></div>{item.source_sentence && <p>{item.source_sentence}</p>}<small>{item.paper_title}</small><button aria-label={`${item.term} 삭제`} onClick={() => { void api.deleteWord(item.id).then(refreshVocabulary).catch((error: Error) => setNotice(error.message)) }}>×</button></article>)}{vocabulary.length === 0 && <p className="empty-note">논문에서 모르는 단어를 눌러보세요.</p>}</div></div>}
                </div>
              </aside>
            </div>
          )}
          {notice && <div className="toast" role="status">{notice}<button onClick={() => setNotice('')} aria-label="닫기">×</button></div>}
        </main>
      )}
    </div>
  )
}

function TranslationPanel({ sentence, translation }: { sentence?: Sentence; translation?: string }) {
  return <div className="panel-section"><p className="eyebrow">IN CONTEXT</p><h2>문장 번역</h2>{sentence && translation ? <><p className="panel-meta">PAGE {sentence.page + 1}</p><div className="sentence-original">{sentence.text}</div><div className="sentence-divider">↓</div><div className="sentence-korean">{translation}</div></> : <div className="empty-panel"><span>“</span><p>문장 선택 모드에서<br />논문의 문장을 눌러보세요.</p></div>}</div>
}

function ResearchPanel({ sources, warnings, summary }: { sources: Source[]; warnings: string[]; summary: string }) {
  return <div className="panel-section"><p className="eyebrow">CONNECTED RESEARCH</p><h2>관련 자료</h2>{summary && <div className="summary-card"><span>논문 맥락 요약</span><p>{summary}</p></div>}{warnings.map((warning) => <p className="source-warning" key={warning}>{warning}</p>)}<div className="source-list">{sources.map((source) => <SourceCard key={source.id} source={source} />)}{sources.length === 0 && <p className="empty-note">수집된 외부 자료가 없습니다. 현재 논문만으로도 질문할 수 있습니다.</p>}</div><p className="source-footnote">외부 논문은 공개된 서지 정보와 초록을 사용합니다. 전문을 수집한 것으로 표시하지 않습니다.</p></div>
}

function SourceCard({ source, excerpt = false }: { source: Source; excerpt?: boolean }) {
  const scope = source.content_scope === 'abstract' ? '초록' : source.content_scope === 'metadata' ? '서지 정보만' : null
  return <article className="source-card"><div className="source-card-top"><span className="source-kind">{sourceLabels[source.kind]}</span>{scope && <span>{scope}</span>}{source.year && <span>{source.year}</span>}{source.page && <span>p. {source.page}</span>}</div><h3>{source.url ? <a href={source.url} target="_blank" rel="noopener noreferrer">{source.title} ↗</a> : source.title}</h3>{excerpt && source.excerpt && <p>{source.excerpt}</p>}</article>
}
