import type {
  AxiomContentBlock,
  AxiomImageBlock,
  AxiomMessage,
  AxiomStreamEvent,
  AxiomTextBlock,
  AxiomToolResult,
  AxiomToolSchema,
  AxiomToolUse,
  AxiomWebSearchResult,
  AxiomWebSearchToolSchema,
} from './types.js'

export type ToolResultBlockParam = AxiomToolResult
export type ToolUseBlockParam = AxiomToolUse
export type BetaToolUseBlock = AxiomToolUse
export type BetaContentBlock = AxiomContentBlock
export type BetaWebSearchTool20250305 = AxiomWebSearchToolSchema

export type Base64ImageSource = AxiomImageBlock['source']

export class APIUserAbortError extends Error {
  constructor(message = 'User aborted the request') {
    super(message)
    this.name = 'APIUserAbortError'
  }
}

export class AnthropicProviderAdapter {
  static toAnthropicContentBlock(block: AxiomContentBlock): Record<string, unknown> {
    if (block.type === 'text') {
      return { type: 'text', text: block.text }
    }
    if (block.type === 'image') {
      return { type: 'image', source: block.source }
    }
    if (block.type === 'tool_use') {
      return {
        type: 'tool_use',
        id: block.id,
        name: block.name,
        input: block.input,
      }
    }
    if (block.type === 'tool_result') {
      return {
        type: 'tool_result',
        tool_use_id: block.tool_use_id,
        content: block.content,
        is_error: block.is_error,
      }
    }
    if (block.type === 'server_tool_use') {
      return {
        type: 'server_tool_use',
        id: block.id,
        name: block.name,
        input: block.input,
      }
    }
    return {
      type: 'web_search_tool_result',
      tool_use_id: block.tool_use_id,
      content: block.content,
    }
  }

  static fromAnthropicContentBlock(block: unknown): AxiomContentBlock {
    const raw = asRecord(block)
    const type = String(raw.type ?? 'text')
    if (type === 'text') {
      return { type: 'text', text: String(raw.text ?? '') }
    }
    if (type === 'image') {
      const source = asRecord(raw.source)
      return {
        type: 'image',
        source: {
          type: 'base64',
          media_type: String(source.media_type ?? 'application/octet-stream'),
          data: String(source.data ?? ''),
        },
      }
    }
    if (type === 'tool_use' || type === 'server_tool_use') {
      return {
        type: type === 'tool_use' ? 'tool_use' : 'server_tool_use',
        id: String(raw.id ?? raw.tool_use_id ?? ''),
        name: String(raw.name ?? ''),
        input: asRecord(raw.input),
      } as AxiomToolUse
    }
    if (type === 'tool_result') {
      return {
        type: 'tool_result',
        tool_use_id: String(raw.tool_use_id ?? ''),
        content: coerceToolResultContent(raw.content),
        is_error: raw.is_error === true,
      }
    }
    if (type === 'web_search_tool_result') {
      return {
        type: 'web_search_tool_result',
        tool_use_id: String(raw.tool_use_id ?? ''),
        content: Array.isArray(raw.content)
          ? raw.content.map(hit => {
              const item = asRecord(hit)
              return {
                title: String(item.title ?? ''),
                url: String(item.url ?? ''),
                page_age: item.page_age == null ? null : String(item.page_age),
              }
            })
          : { error_code: String(asRecord(raw.content).error_code ?? 'unknown') },
      }
    }
    return { type: 'text', text: JSON.stringify(raw) }
  }

  static toAnthropicMessage(message: AxiomMessage): Record<string, unknown> {
    return {
      role: message.role,
      content: typeof message.content === 'string'
        ? message.content
        : message.content.map(block => this.toAnthropicContentBlock(block)),
    }
  }

  static fromAnthropicMessage(message: unknown): AxiomMessage {
    const raw = asRecord(message)
    const content = raw.content
    return {
      role: String(raw.role ?? 'assistant') as AxiomMessage['role'],
      id: raw.id == null ? undefined : String(raw.id),
      content: typeof content === 'string'
        ? content
        : Array.isArray(content)
          ? content.map(block => this.fromAnthropicContentBlock(block))
          : [],
    }
  }

  static toAnthropicToolSchema(schema: AxiomToolSchema): Record<string, unknown> {
    return { ...schema }
  }

  static fromAnthropicStreamEvent(event: unknown): AxiomStreamEvent {
    const raw = asRecord(event)
    const type = String(raw.type ?? 'error')
    if (type === 'content_block_start') {
      return {
        type,
        index: Number(raw.index ?? 0),
        content_block: this.fromAnthropicContentBlock(raw.content_block),
      }
    }
    if (type === 'content_block_delta') {
      return {
        type,
        index: Number(raw.index ?? 0),
        delta: asRecord(raw.delta),
      }
    }
    if (type === 'content_block_stop') {
      return { type, index: Number(raw.index ?? 0) }
    }
    if (type === 'message_start') {
      return { type, message: this.fromAnthropicMessage(raw.message) }
    }
    if (type === 'message_stop') {
      return { type }
    }
    return {
      type: 'error',
      error: {
        type: String(asRecord(raw.error).type ?? 'provider_error'),
        message: String(asRecord(raw.error).message ?? JSON.stringify(raw)),
      },
    }
  }

  static toToolResultBlock(content: string, toolUseID: string, isError = false): AxiomToolResult {
    return {
      type: 'tool_result',
      tool_use_id: toolUseID,
      content,
      is_error: isError || undefined,
    }
  }

  static webSearchTool(input: {
    allowed_domains?: string[]
    blocked_domains?: string[]
    max_uses?: number
  }): AxiomWebSearchToolSchema {
    return {
      type: 'web_search_20250305',
      name: 'web_search',
      allowed_domains: input.allowed_domains,
      blocked_domains: input.blocked_domains,
      max_uses: input.max_uses ?? 8,
    }
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function coerceToolResultContent(value: unknown): string | AxiomTextBlock[] {
  if (typeof value === 'string') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map(item => {
      const raw = asRecord(item)
      return { type: 'text', text: String(raw.text ?? '') }
    })
  }
  return JSON.stringify(value ?? '')
}

export function toAxiomContentBlock(block: unknown): AxiomContentBlock {
  return AnthropicProviderAdapter.fromAnthropicContentBlock(block)
}

export function fromAxiomContentBlock(block: AxiomContentBlock): Record<string, unknown> {
  return AnthropicProviderAdapter.toAnthropicContentBlock(block)
}

export function isAxiomWebSearchResult(block: AxiomContentBlock): block is AxiomWebSearchResult {
  return block.type === 'web_search_tool_result'
}
