import { useEffect, useState } from 'react'
import { useSessionStore } from '@/stores/sessionStore'
import { Plus, MessageSquare, Trash2, Edit2, Database } from 'lucide-react'
import { deleteSession, updateSession } from '@/api/sessions'
import { getKnowledgeBases } from '@/api/knowledgeBases'
import type { KnowledgeBase } from '@/types/knowledgeBase'

export const Sidebar = () => {
  const { sessions, currentSessionId, isLoading, loadSessions, setCurrentSession, createSession } = useSessionStore()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingName, setEditingName] = useState('')
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [selectedKbId, setSelectedKbId] = useState<string>('')

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    getKnowledgeBases()
      .then((data) => setKnowledgeBases(data.items))
      .catch(() => console.warn('加载知识库列表失败'))
  }, [])

  const handleNewSession = async () => {
    await createSession('新会话', selectedKbId ? [selectedKbId] : [])
  }

  const handleDelete = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm('确定删除此会话？')) {
      const optimisticSessions = sessions.filter(s => s.session_id !== sessionId)

      const remainingSession = optimisticSessions.find(s => s.session_id !== sessionId)
      if (currentSessionId === sessionId) {
        setCurrentSession(remainingSession ? remainingSession.session_id : '')
      }

      try {
        await deleteSession(sessionId)
        loadSessions()
      } catch (error) {
        console.error('删除会话失败:', error)
        loadSessions()
      }
    }
  }

  const handleRename = async (sessionId: string) => {
    try {
      await updateSession(sessionId, { name: editingName })
      setEditingId(null)
      loadSessions()
    } catch (error) {
      console.error('重命名失败:', error)
    }
  }

  return (
    <div className="h-full bg-white border-r flex flex-col">
      <div className="p-4 border-b space-y-3">
        {knowledgeBases.length > 0 && (
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-gray-500 mb-1.5">
              <Database size={12} />
              知识库
            </label>
            <select
              value={selectedKbId}
              onChange={(e) => setSelectedKbId(e.target.value)}
              className="w-full text-sm border border-gray-300 rounded-lg px-2.5 py-1.5 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              <option value="">不选择</option>
              {knowledgeBases.map((kb) => (
                <option key={kb.kb_id} value={kb.kb_id}>
                  {kb.name} ({kb.document_count ?? 0} 文档)
                </option>
              ))}
            </select>
          </div>
        )}
        <button
          onClick={handleNewSession}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus size={16} />
          新建会话
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {isLoading ? (
          <div className="text-center text-gray-500 py-4">加载中...</div>
        ) : (
          sessions.map((session) => (
            <div
              key={session.session_id}
              onClick={() => setCurrentSession(session.session_id)}
              className={`group flex items-center gap-3 px-3 py-2 rounded-lg mb-1 cursor-pointer ${
                currentSessionId === session.session_id
                  ? 'bg-blue-100 text-blue-900'
                  : 'hover:bg-gray-100'
              }`}
            >
              <MessageSquare size={16} />

              {editingId === session.session_id ? (
                <input
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  onBlur={() => handleRename(session.session_id)}
                  onKeyDown={(e) => e.key === 'Enter' && handleRename(session.session_id)}
                  className="flex-1 bg-white px-2 py-0.5 text-sm border rounded"
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <span className="flex-1 truncate">{session.name}</span>
              )}

              <div className="hidden group-hover:flex gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setEditingId(session.session_id)
                    setEditingName(session.name)
                  }}
                  className="p-1 hover:bg-gray-200 rounded"
                >
                  <Edit2 size={14} />
                </button>
                <button
                  onClick={(e) => handleDelete(session.session_id, e)}
                  className="p-1 hover:bg-red-100 text-red-600 rounded"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
