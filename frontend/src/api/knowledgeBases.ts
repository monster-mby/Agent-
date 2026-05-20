import { apiClient } from './client'
import type { KnowledgeBase, PaginatedResponse } from '@/types/knowledgeBase'

export const getKnowledgeBases = async () => {
  const response = await apiClient.get<PaginatedResponse<KnowledgeBase>>('/knowledge-bases', {
    params: { limit: 100 },
  })
  return response.data
}
