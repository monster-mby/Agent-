import { apiClient } from './client'
import type { Session, CreateSessionRequest } from '@/types/session'
import type { Message, PaginatedResponse } from '@/types/message'

export const getSessions = async (limit = 50, offset = 0) => {
  const response = await apiClient.get<PaginatedResponse<Session>>('/sessions', {
    params: { limit, offset },
  })
  return response.data
}

export const createSession = async (data: CreateSessionRequest) => {
  const response = await apiClient.post<Session>('/sessions', data)
  return response.data
}

export const getSessionById = async (sessionId: string) => {
  const response = await apiClient.get<Session>(`/sessions/${sessionId}`)
  return response.data
}

export const deleteSession = async (sessionId: string) => {
  await apiClient.delete(`/sessions/${sessionId}`)
}

export const updateSession = async (sessionId: string, data: { name: string }) => {
  const response = await apiClient.put<Session>(`/sessions/${sessionId}`, data)
  return response.data
}

export const getMessageHistory = async (sessionId: string, limit = 50) => {
  const response = await apiClient.get<Message[]>(`/sessions/${sessionId}/history`, {
    params: { limit },
  })
  return response.data
}
