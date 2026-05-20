export type RuleCategory = 'general' | 'format' | 'style' | 'content';

export interface Rule {
  rule_id: string;
  session_id: string;
  content: string;
  priority: number;
  category: RuleCategory;
  enabled: boolean;
  created_at: string;
  updated_at?: string;
}

export interface CreateRuleRequest {
  content: string;
  priority: number;
  category: RuleCategory;
}
