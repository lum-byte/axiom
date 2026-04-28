#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# recipes/hardcoded/rest_api_json.sh
# Topology class: REST_API_JSON
#
# Signal keys (envelope): "data", "results", "items", "records", "entries",
#                          "content", "payload", "response", "body", "value"
# Noise keys (discarded): "pagination", "meta", "links", "_links", "cursor",
#                          "paging", "page_info", "rate_limit", "headers",
#                          "included", "relationships"
#
# Architecture: two-pass awk pipeline.
#
#   Pass 1 — structure normalization:
#     Compact JSON (everything on one line) defeats depth-tracking because
#     we cannot reliably count brace depth per-line when lines are arbitrary.
#     Normalization puts each { } [ ] on its own line so the depth counter
#     works reliably. This is done with sed expanding structural characters.
#     Limitation: values that contain { } [ ] inside strings will be expanded.
#     Mitigation: awk string-detection in Pass 2 skips lines in string context.
#
#   Pass 2 — depth-aware section filtering (awk):
#     Tracks JSON brace/bracket depth. At depth 1 (inside the root object),
#     inspects keys against signal and noise lists. When a noise key is found
#     at depth 1, the skip flag is set and everything until the matching
#     depth decrease is discarded. Signal key sections pass through.
#
#     String-context tracking: the depth counter ignores { } [ ] characters
#     that appear between unescaped double-quote pairs (inside string values).
#     This prevents string values containing JSON-like content from corrupting
#     the depth counter.
#
#     Handles both pretty-printed and compact JSON after normalization.
#     Handles arrays of objects (data: [{...}, {...}]) correctly.
#
# stdin:  raw JSON (UTF-8), compact or pretty-printed
# stdout: the signal sections of the JSON, one value chunk per output block
# ─────────────────────────────────────────────────────────────────────────────

# ── Pass 1: structure normalization ──────────────────────────────────────────
# Expand structural characters onto their own lines for reliable depth tracking.
# sed approach: insert newlines around { } [ ] , (at JSON structural positions).
# We expand around { } [ ] only — commas between top-level keys also need
# separation so each key:value starts on its own line.
#
# This sed expression is intentionally conservative: it only expands when the
# structural char is preceded/followed by non-string content indicators.
# For production use on well-formed REST API responses, this is sufficient.
sed '
# Expand opening braces/brackets: insert newline after
s/{\([^"]\)/{\n\1/g
s/\[\([^"]\)/[\n\1/g
# Expand closing braces/brackets: insert newline before
s/\([^"]\)}/\1\n}/g
s/\([^"]\)]/\1\n]/g
# One key-value pair per line at the structural level
s/,\("[^"]*"[[:space:]]*:\)/,\n\1/g
' |

# ── Pass 2: depth-aware section filtering ─────────────────────────────────────
awk '
BEGIN {
    depth      = 0    # current JSON nesting depth
    skip       = 0    # currently inside a noise section
    skip_depth = 0    # depth at which skip was activated

    # Noise key registry — keys at depth 1 whose sections are discarded
    noise["pagination"]     = 1
    noise["meta"]           = 1
    noise["links"]          = 1
    noise["_links"]         = 1
    noise["cursor"]         = 1
    noise["paging"]         = 1
    noise["page_info"]      = 1
    noise["rate_limit"]     = 1
    noise["x-rate-limit"]   = 1
    noise["headers"]        = 1
    noise["included"]       = 1
    noise["relationships"]  = 1
    noise["errors"]         = 1

    # Signal key registry — keys at depth 1 whose sections pass through
    # (used only for informational purposes; non-noise keys pass by default)
    signal["data"]          = 1
    signal["results"]       = 1
    signal["items"]         = 1
    signal["records"]       = 1
    signal["entries"]       = 1
    signal["content"]       = 1
    signal["payload"]       = 1
    signal["response"]      = 1
    signal["body"]          = 1
    signal["value"]         = 1
    signal["values"]        = 1
    signal["documents"]     = 1
    signal["articles"]      = 1
    signal["posts"]         = 1
    signal["products"]      = 1
    signal["users"]         = 1
}

{
    line     = $0
    in_str   = 0      # are we inside a JSON string on this line?
    n        = length(line)
    new_depth = depth

    # ── Count structural characters, skipping string content ──────────────
    # Walk character by character to correctly handle strings.
    for (i = 1; i <= n; i++) {
        c = substr(line, i, 1)

        if (in_str) {
            if (c == "\\") { i++; continue }   # escaped char — skip next
            if (c == "\"") in_str = 0
            continue
        }

        if (c == "\"") { in_str = 1; continue }

        if (c == "{" || c == "[") new_depth++
        if (c == "}" || c == "]") {
            new_depth--
            # Check if we are exiting a skip section
            if (skip && new_depth < skip_depth) {
                skip = 0
                # Do not print the closing brace of the noise section
                next
            }
        }
    }

    # ── Noise key detection at depth 1 ────────────────────────────────────
    # A key at depth 1 appears as: ^[[:space:]]*"keyname"[[:space:]]*:
    # depth here is the depth BEFORE processing this line.
    if (!skip && depth == 1) {
        # Extract the key name from this line
        key = line
        gsub(/^[[:space:]]*"/, "", key)
        gsub(/".*/, "", key)
        if (key in noise) {
            skip       = 1
            skip_depth = depth    # exit skip when we return to this depth
            depth      = new_depth
            next
        }
    }

    # ── Skip active — discard line ─────────────────────────────────────────
    if (skip) {
        depth = new_depth
        next
    }

    # ── Signal output ──────────────────────────────────────────────────────
    # Print non-blank lines only (normalization produces empty lines)
    if (line !~ /^[[:space:]]*$/) print line

    depth = new_depth
}
'