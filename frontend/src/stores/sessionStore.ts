import { create } from 'zustand'
import type { Session } from '@/types/session'
import type { Message } from '@/types/message'
import { getSessions, createSession as apiCreateSession, getMessageHistory } from '@/api/sessions'

interface SessionState {
  sessions: Session[]
  currentSessionId: string | null
  messages: Message[]
  isLoading: boolean

  loadSessions: () => Promise<void>
  setCurrentSession: (id: string) => Promise<void>
  createSession: (name: string, knowledgeBaseIds?: string[]) => Promise<void>
  addMessage: (message: Message) => void
  clearMessages: () => void
}

export const useSessionStore = create<SessionState>((set) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  isLoading: false,

  loadSessions: async () => {
    set({ isLoading: true })
    try {
      const data = await getSessions()
      set({ sessions: data.items })
    } catch (error: any) {
      console.error('加载会话列表失败:', error)
      alert('加载会话失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      set({ isLoading: false })
    }
  },

  setCurrentSession: async (id) => {
    set({ currentSessionId: id, isLoading: true })
    try {
      const messages = await getMessageHistory(id)
      set({ messages })
    } catch (error: any) {
      console.error('加载消息历史失败:', error)
      alert('加载消息历史失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      set({ isLoading: false })
    }
  },

  createSession: async (name, knowledgeBaseIds) => {
    try {
      const session = await apiCreateSession({
        name,
        ...(knowledgeBaseIds?.length ? { knowledge_base_ids: knowledgeBaseIds } : {}),
      })
      set((state) => ({
        sessions: [session, ...state.sessions],
        currentSessionId: session.session_id,
        messages: [],
      }))
    } catch (error: any) {
      console.error('创建会话失败:', error)
      alert('创建会话失败: ' + (error.response?.data?.detail || error.message))
    }
  },

  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message],
  })),

  clearMessages: () => set({ messages: [] }),
}))
