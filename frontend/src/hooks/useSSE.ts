import { useState, useCallback, useRef } from 'react'
import type { AnswerChunkData, DoneData } from '@/types/api'

interface StartStreamingParams {
  sessionId: string
  query: string
  onChunk: (chunk: string) => void
  onDone: (data: DoneData) => void
  onError: (error: string) => void
}

interface UseSSEReturn {
  isStreaming: boolean
  startStreaming: (params: StartStreamingParams) => Promise<void>
  abort: () => void
}

export const useSSE = (): UseSSEReturn => {
  const [isStreaming, setIsStreaming] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)

  const startStreaming = useCallback(async ({
    sessionId,
    query,
    onChunk,
    onDone,
    onError,
  }: StartStreamingParams) => {
    setIsStreaming(true)
    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      const response = await fetch(`/api/v1/sessions/${sessionId}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': import.meta.env.VITE_API_KEY || '',
        },
        body: JSON.stringify({ query, stream: true }),
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let eventType = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim()
            try {
              const data = JSON.parse(dataStr)

              if (eventType === 'answer_chunk') {
                onChunk((data as AnswerChunkData).chunk)
              } else if (eventType === 'done') {
                onDone(data as DoneData)
                setIsStreaming(false)
              } else if (eventType === 'error') {
                onError((data as any).error)
                setIsStreaming(false)
              }
            } catch (parseError) {
              console.error('SSE 数据解析失败:', parseError)
            }
            eventType = ''
          }
        }
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        onError(error.message)
        setIsStreaming(false)
      }
    }
  }, [])

  const abort = useCallback(() => {
    abortControllerRef.current?.abort()
    setIsStreaming(false)
  }, [])

  return { isStreaming, startStreaming, abort }
}
