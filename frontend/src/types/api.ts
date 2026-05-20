export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
}

export interface SSEEvent {
  id?: string;
  event: 'node_output' | 'answer_chunk' | 'done' | 'error';
  data: string;
}

export interface AnswerChunkData {
  chunk: string;
  request_id: string;
  session_id: string;
}

export interface DoneData {
  status: string;
  elapsed_ms: number;
  total_tokens: number;
  request_id: string;
  session_id: string;
}

export interface ErrorData {
  error: string;
  request_id: string;
  session_id: string;
}
