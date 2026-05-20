export interface Session {
  session_id: string;
  user_id: string;
  name: string;
  knowledge_base_ids: string[];
  langgraph_thread_id: string;
  status: 'active' | 'archived' | 'deleted';
  rules: string[];
  created_at: string;
  updated_at: string;
}

export interface CreateSessionRequest {
  name: string;
  knowledge_base_ids?: string[];
}
