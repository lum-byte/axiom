export type AlpineStripArtifactKind =
  | 'raw_html'
  | 'rendered_html'
  | 'markdown'
  | 'metadata'
  | 'plain_text'

export type AlpineStripRequest = {
  tool: 'AlpineStripTool'
  artifact_kind: AlpineStripArtifactKind
  url?: string
  query?: string
  topology_class?: string
  input?: string
  input_path?: string
  output_path?: string
  slot_idx?: number
  max_output_ratio?: number
}

export type AlpineStripResponse = {
  ok: boolean
  status: 'planned' | 'ok' | 'error'
  native: 'alpine_strip/tool_strip_accelerator.c'
  request: AlpineStripRequest
  queue_line?: string
}

export const AlpineStripTool = {
  name: 'AlpineStripTool',
  description:
    'Native AXIOM strip accelerator for TAG-routed tool snapshots and offline recipe batches.',
  input_schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      artifact_kind: {
        type: 'string',
        enum: ['raw_html', 'rendered_html', 'markdown', 'metadata', 'plain_text'],
      },
      url: { type: 'string' },
      query: { type: 'string' },
      topology_class: { type: 'string' },
      input: { type: 'string' },
      input_path: { type: 'string' },
      output_path: { type: 'string' },
      slot_idx: { type: 'number' },
      max_output_ratio: { type: 'number' },
    },
    required: ['artifact_kind'],
  },
}

export function normalizeAlpineStripRequest(input: Record<string, unknown>): AlpineStripRequest {
  const artifactKind = typeof input.artifact_kind === 'string'
    ? input.artifact_kind as AlpineStripArtifactKind
    : 'raw_html'
  return {
    tool: 'AlpineStripTool',
    artifact_kind: artifactKind,
    url: stringOrUndefined(input.url),
    query: stringOrUndefined(input.query),
    topology_class: stringOrUndefined(input.topology_class) ?? 'GENERIC_HTML',
    input: stringOrUndefined(input.input),
    input_path: stringOrUndefined(input.input_path),
    output_path: stringOrUndefined(input.output_path),
    slot_idx: typeof input.slot_idx === 'number' ? input.slot_idx : 0,
    max_output_ratio: typeof input.max_output_ratio === 'number' ? input.max_output_ratio : undefined,
  }
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}
