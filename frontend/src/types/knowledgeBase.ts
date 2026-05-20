export interface KnowledgeBase {
  kb_id: string
  name: string
  description: string
  status: string
  document_count: number
  created_at: string | null
  updated_at: string | null
}

/** 通用分页响应（与 api.ts 中的一致） */
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  size: number
}
