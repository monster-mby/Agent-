export interface Message {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  retrieval_sources?: RetrievalSource[]
  applied_rules?: string[]
}

export interface RetrievalSource {
  document_id: string
  content: string
  score: number
}

export interface SendMessageRequest {
  query: string
  knowledge_base_ids?: string[]
  stream: boolean
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  size: number
}
