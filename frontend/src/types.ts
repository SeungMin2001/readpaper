export type Word = {
  id: string
  text: string
  page: number
  x0: number
  y0: number
  x1: number
  y1: number
  sentence_id: string
}

export type PaperPage = {
  index: number
  width: number
  height: number
  words: Word[]
}

export type Sentence = {
  id: string
  page: number
  text: string
  word_ids: string[]
}

export type TranslationUnit = {
  word_ids: string[]
  original: string
  meaning: string
}

export type SentenceTranslation = {
  translation: string
  units: TranslationUnit[]
}

export type Source = {
  id: string
  kind: 'paper' | 'referenced' | 'related' | 'citing' | 'recent' | 'background'
  title: string
  content_scope?: 'full_text' | 'abstract' | 'metadata' | 'background'
  openalex_id?: string | null
  doi?: string | null
  publication_date?: string | null
  url: string | null
  year?: number | null
  authors?: string[]
  page?: number
  excerpt?: string
}

export type Paper = {
  id: string
  title: string
  status: 'processing' | 'ready' | 'failed'
  progress: number
  error: string | null
  pages?: PaperPage[]
  sentences?: Sentence[]
  translations?: Record<string, SentenceTranslation>
  profile?: { summary: string; keywords: string[] }
  sources?: Source[]
  source_warnings?: string[]
}

export type VocabularyItem = {
  id: number
  paper_title: string
  term: string
  meaning: string
  source_sentence: string
  created_at: number
}

export type Answer = {
  answer: string
  citations: string[]
  sources: Source[]
}
