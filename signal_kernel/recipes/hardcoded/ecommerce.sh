#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# recipes/hardcoded/ecommerce.sh
# Topology class: ECOMMERCE_PRODUCT
#
# Signal attributes:  data-product, data-product-id, data-sku, data-name,
#                     data-price, data-price-*, data-currency, data-category,
#                     data-brand, data-in-stock, data-stock-*, data-variant,
#                     data-color, data-size, data-rating, data-review-count
#
# Noise attributes:   data-analytics, data-analytics-*, data-tracking,
#                     data-tracking-*, data-gtm, data-gtm-*, data-beacon,
#                     data-pixel, data-impression
#
# Architecture: three-pass awk pipeline.
#
#   The core challenge: signal and noise co-exist on the SAME HTML element.
#   A product container carries data-product (signal) and data-analytics
#   (noise) and data-gtm (noise) on the same tag. You cannot keep or drop
#   entire elements — you must operate at the attribute level. Additionally,
#   prose content (description text) lives INSIDE tagged elements, so tag
#   lines must both: (a) emit their data-* attribute values, and (b) emit
#   any prose text content within them.
#
#   Pass 1 — noise zone elimination + multi-line tag linearization (awk):
#     Eliminates noise zones that are pure non-product content:
#       <script>, <style>, <nav>, <aside>, <footer>, <noscript>
#     State machine: skip=1 on opening tag, skip=0 on closing tag.
#     Tag name extracted by finding first < and reading alpha chars after —
#     NOT sub(/.*</) which is greedy and breaks on same-line open+close tags.
#     After elimination, linearizes multi-line tags (one complete tag → one
#     output line) for reliable per-line attribute processing downstream.
#
#   Pass 2 — noise attribute erasure (awk):
#     while(match(pat)) loops erase all noise data-* attribute families.
#     Drops tracking-only containers by class pattern.
#
#   Pass 3 — signal extraction (awk):
#     Each line is either a tag line or a bare prose line.
#     Tag lines: (a) extract data-* signal attribute values as "label: value"
#                (b) strip all HTML tags and emit any remaining prose content.
#     Bare prose lines: strip remaining tags, decode entities, emit.
#     This dual-channel approach captures both structured product attributes
#     and free-text description paragraphs from a single awk pass.
#
# stdin:  raw HTML (UTF-8)
# stdout: "label: value" attribute pairs interleaved with prose description
# ─────────────────────────────────────────────────────────────────────────────

# ── Pass 1: noise zone elimination + tag linearization ───────────────────────
awk '
BEGIN { skip = 0; skip_tag = ""; buf = "" }

!skip && /<(script|style|nav|aside|footer|noscript)[[:space:]>]/ {
    p    = index($0, "<")
    rest = substr($0, p + 1)
    tag  = rest; gsub(/[^a-zA-Z].*/, "", tag)
    skip_tag = tolower(tag)
    if (index(tolower($0), "</" skip_tag ">")) next
    skip = 1; next
}
skip {
    if (index(tolower($0), "</" skip_tag ">")) { skip = 0; skip_tag = "" }
    next
}
{
    if (buf != "") {
        buf = buf " " $0
        if (index(buf, ">")) { print buf; buf = "" }
        next
    }
    if ($0 ~ /<[^>]*$/) { buf = $0 }
    else                 { print }
}
END { if (buf != "") print buf }
' |

# ── Pass 2: noise attribute erasure ──────────────────────────────────────────
awk '
{
    pat = "data-analytics[a-zA-Z-]*=\"[^\"]*\""
    while (match($0, pat)) $0 = substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH)
    pat = "data-tracking[a-zA-Z-]*=\"[^\"]*\""
    while (match($0, pat)) $0 = substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH)
    pat = "data-gtm[a-zA-Z-]*=\"[^\"]*\""
    while (match($0, pat)) $0 = substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH)
    pat = "data-beacon[a-zA-Z-]*=\"[^\"]*\""
    while (match($0, pat)) $0 = substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH)
    pat = "data-pixel[a-zA-Z-]*=\"[^\"]*\""
    while (match($0, pat)) $0 = substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH)
    pat = "data-impression[a-zA-Z-]*=\"[^\"]*\""
    while (match($0, pat)) $0 = substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH)
    gsub(/  +/, " ")
    if ($0 ~ /class="[^"]*tracking-pixel/)   next
    if ($0 ~ /class="[^"]*analytics-beacon/) next
    print
}
' |

# ── Pass 3: signal extraction + prose capture ─────────────────────────────────
awk '
BEGIN {
    n = split("data-product,data-product-id,data-sku,data-name,data-brand,data-category,data-price,data-price-original,data-price-discount,data-price-currency,data-currency,data-in-stock,data-stock-count,data-variant,data-color,data-size,data-rating,data-review-count", attrs, ",")
}

# ── Tag line: extract both attribute values AND prose content ─────────────────
/^[[:space:]]*<[a-zA-Z!\/]/ {
    line = $0

    # Sub-pass A: extract signal attribute values
    for (i = 1; i <= n; i++) {
        attr = attrs[i]; val = ""
        dq = attr "=\""
        p  = index(line, dq)
        if (p) {
            rest = substr(line, p + length(dq))
            q    = index(rest, "\"")
            if (q) val = substr(rest, 1, q-1)
        }
        if (!val) {
            sq = attr "=\047"
            p  = index(line, sq)
            if (p) {
                rest = substr(line, p + length(sq))
                for (ci = 1; ci <= length(rest); ci++) {
                    if (substr(rest,ci,1) == "\047") { val = substr(rest,1,ci-1); break }
                }
            }
        }
        if (val != "") {
            label = attr; sub(/^data-/, "", label); gsub(/-/, " ", label)
            print label ": " val
        }
    }

    # Sub-pass B: extract prose text content from within the tag line
    # Strip all HTML tags; what remains is text between/after tags on this line
    while (match(line, /<[^>]*>/)) line = substr(line,1,RSTART-1) substr(line,RSTART+RLENGTH)
    gsub(/&amp;/,  "\\&", line); gsub(/&lt;/, "<", line)
    gsub(/&gt;/,   ">",   line); gsub(/&nbsp;/, " ", line)
    gsub(/[[:space:]]+/, " ", line); gsub(/^ | $/, "", line)
    if (length(line)) print line
    next
}

# ── Bare prose line: strip tags, decode entities, emit ───────────────────────
{
    while (match($0, /<[^>]*>/)) $0 = substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH)
    gsub(/&amp;/,  "\\&"); gsub(/&lt;/, "<")
    gsub(/&gt;/,   ">");   gsub(/&nbsp;/, " "); gsub(/&quot;/, "\"")
    gsub(/[[:space:]]+/, " "); gsub(/^ | $/, "")
    if (length($0)) print
}
'