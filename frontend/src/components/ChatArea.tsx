import { useState } from 'react'
import { useSessionStore } from '@/stores/sessionStore'
import { useSSE } from '@/hooks/useSSE'
import { MessageBubble } from './MessageBubble'
import { Send, Loader2 } from 'lucide-react'

export const ChatArea = () => {
  const { currentSessionId, messages, isLoading } = useSessionStore()
  const { isStreaming, startStreaming, abort } = useSSE()
  const [input, setInput] = useState('')

  const handleSend = async () => {
    if (!input.trim() || !currentSessionId) return

    const userMessage = {
      message_id: Date.now().toString(),
      role: 'user' as const,
      content: input,
      created_at: new Date().toISOString(),
    }

    useSessionStore.getState().addMessage(userMessage)
    setInput('')

    const aiMessageId = (Date.now() + 1).toString()
    const aiMessage = {
      message_id: aiMessageId,
      role: 'assistant' as const,
      content: '',
      created_at: new Date().toISOString(),
    }
    useSessionStore.getState().addMessage(aiMessage)

    let fullContent = ''
    await startStreaming({
      sessionId: currentSessionId,
      query: input,
      onChunk: (chunk) => {
        fullContent += chunk
        const state = useSessionStore.getState()
        const updatedMessages = state.messages.map((msg: any) =>
          msg.message_id === aiMessageId ? { ...msg, content: fullContent } : msg
        )
        state.messages = updatedMessages
      },
      onDone: () => {
        console.log('流式输出完成')
      },
      onError: (error) => {
        console.error('流式输出错误:', error)
      },
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  if (!currentSessionId) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-50">
        <p className="text-gray-500">选择一个会话开始对话</p>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col bg-gray-50">
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="animate-spin" size={32} />
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.message_id} message={msg} />
          ))
        )}
      </div>

      <div className="border-t bg-white p-4">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入消息... (Enter 发送，Shift+Enter 换行)"
            className="flex-1 px-4 py-2 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={1}
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button
              onClick={abort}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              停止
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              <Send size={20} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
