import type { ActivityEvent, Answer, GlossPreparation, Paper, VocabularyItem, WordGloss } from './types'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { credentials: 'same-origin', ...options })
  if (!response.ok) {
    let message = `요청에 실패했습니다 (${response.status}).`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch { /* Keep the HTTP error. */ }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  me: () => request<{ username: string }>('/api/auth/me'),
  login: (username: string, password: string) => request<{ username: string }>('/api/auth/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  }),
  logout: () => request<{ ok: boolean }>('/api/auth/logout', { method: 'POST' }),
  upload: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<{ paper_id: string; status: string; cached: boolean }>('/api/papers/upload', {
      method: 'POST', body,
    })
  },
  paper: (id: string) => request<Paper>(`/api/papers/${id}`),
  glossPreparation: (id: string) => request<GlossPreparation>(`/api/papers/${id}/gloss-preparation`),
  retryGlossPreparation: (id: string) => request<GlossPreparation>(`/api/papers/${id}/gloss-preparation`, { method: 'POST' }),
  retry: (id: string) => request<{ status: string }>(`/api/papers/${id}/retry`, { method: 'POST' }),
  translateWord: (paperId: string, wordId: string) => request<WordGloss & { cached: boolean; elapsed_ms: number; activity?: ActivityEvent[] }>(`/api/papers/${paperId}/translate-word`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ word_id: wordId }),
  }),
  ask: (id: string, question: string) => request<Answer>(`/api/papers/${id}/ask`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  }),
  vocabulary: () => request<VocabularyItem[]>('/api/vocabulary'),
  saveWord: (word: Omit<VocabularyItem, 'id' | 'created_at'>) => request<{ ok: boolean }>('/api/vocabulary', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(word),
  }),
  deleteWord: (id: number) => request<{ ok: boolean }>(`/api/vocabulary/${id}`, { method: 'DELETE' }),
}
