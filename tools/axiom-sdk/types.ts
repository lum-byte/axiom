export type AxiomRole = 'user' | 'assistant' | 'system' | 'tool'

export type AxiomTextBlock = {
  type: 'text'
  text: string
}

export type AxiomImageBlock = {
  type: 'image'
  source: {
    type: 'base64'
    media_type: string
    data: string
  }
}

export type AxiomToolUse = {
  type: 'tool_use'
  id: string
  name: string
  input: Record<string, unknown>
}

export type AxiomToolResult = {
  type: 'tool_result'
  tool_use_id: string
  content: string | AxiomTextBlock[]
  is_error?: boolean
}

export type AxiomServerToolUse = {
  type: 'server_tool_use'
  id: string
  name: string
  input?: Record<string, unknown>
}

export type AxiomWebSearchHit = {
  title: string
  url: string
  page_age?: string | null
}

export type AxiomWebSearchResult = {
  type: 'web_search_tool_result'
  tool_use_id: string
  content: AxiomWebSearchHit[] | { error_code: string }
}

export type AxiomContentBlock =
  | AxiomTextBlock
  | AxiomImageBlock
  | AxiomToolUse
  | AxiomToolResult
  | AxiomServerToolUse
  | AxiomWebSearchResult

export type AxiomMessage = {
  role: AxiomRole
  content: string | AxiomContentBlock[]
  id?: string
}

export type AxiomToolSchema = {
  name: string
  description?: string
  input_schema?: Record<string, unknown>
  type?: string
  [key: string]: unknown
}

export type AxiomWebSearchToolSchema = AxiomToolSchema & {
  type: 'web_search_20250305'
  name: 'web_search'
  allowed_domains?: string[]
  blocked_domains?: string[]
  max_uses?: number
}

export type AxiomStreamEvent =
  | { type: 'message_start'; message: AxiomMessage }
  | { type: 'content_block_start'; index: number; content_block: AxiomContentBlock }
  | { type: 'content_block_delta'; index: number; delta: Record<string, unknown> }
  | { type: 'content_block_stop'; index: number }
  | { type: 'message_stop' }
  | { type: 'error'; error: { type: string; message: string } }

export type AxiomProviderMessage = {
  type: 'assistant' | 'user' | 'stream_event' | 'progress' | string
  message?: AxiomMessage
  event?: AxiomStreamEvent | Record<string, unknown>
}

export type AxiomPermissionBehavior = 'allow' | 'deny' | 'ask' | 'passthrough'

export type AxiomPermissionDecision = {
  behavior: AxiomPermissionBehavior
  message?: string
  updatedInput?: unknown
  suggestions?: unknown[]
}

export type AxiomToolExecution = {
  ok: boolean
  toolName: string
  invocationId: string
  durationMs: number
  data?: unknown
  structuredOutput?: unknown
  resultBlock?: AxiomToolResult
  error?: {
    type: string
    message: string
  }
}

export type AxiomRegisteredTool = {
  name: string
  adapterKind: string
  permissionClass: string
  sourcePath: string
  status: 'ready' | 'missing_deps' | 'disabled' | 'error'
  dependencies: string[]
}

export type AxiomStripArtifactKind =
  | 'raw_html'
  | 'rendered_html'
  | 'markdown'
  | 'metadata'
  | 'plain_text'

export type AxiomStripRequest = {
  tool: 'AlpineStripTool'
  artifact_kind: AxiomStripArtifactKind
  url?: string
  query?: string
  topology_class?: string
  input?: string
  input_path?: string
  output_path?: string
  slot_idx?: number
  max_output_ratio?: number
}

export type AxiomStripPlan = {
  native: 'alpine_strip/tool_strip_accelerator.c'
  request: AxiomStripRequest
  queue_line?: string
}
