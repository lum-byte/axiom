#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# recipes/hardcoded/saas_docs.sh
# Topology class: SAAS_DOCS
#
# Signal zone: <main>
# Noise stripped: <nav> <aside> <footer> <header> <script> <style>
#                 Elements with sidebar/toc/menu in class attribute.
#                 HTML comments.
#
# Code block handling:
#   <pre> and <code> content is SIGNAL in documentation pages — it is the
#   actual API usage examples that LLMs need for accurate code generation.
#   These blocks are preserved with their internal whitespace intact.
#   Only the structural tags (<pre>, <code>) are removed — the content stays.
#
#   The critical difference from news_article.sh:
#     Regular prose: whitespace collapsed (multiple spaces → one)
#     Code blocks:   leading whitespace PRESERVED (indentation is syntax)
#
#   Implementation: awk tags code lines with a sentinel prefix (CODEBLOCK:)
#   before piping to sed. A final sed pass strips the sentinel from output,
#   having used its presence to select the correct whitespace treatment.
#
# Architecture: four-pass pipeline.
#
#   Pass 1 — comment erasure (sed)
#   Pass 2 — zone + code-block extraction (awk state machine)
#             Emits normal lines as-is.
#             Emits code lines prefixed with \x02 (STX, non-printing sentinel).
#             STX is chosen because it cannot appear in valid UTF-8 HTML content.
#   Pass 3 — tag removal (sed)
#             Applied uniformly — removes <pre>, <code>, and all other tags.
#             The sentinel prefix protects code line identity through this pass.
#   Pass 4 — differential whitespace + sentinel removal (awk)
#             Lines starting with STX: strip STX, preserve leading whitespace,
#             strip trailing whitespace only.
#             All other lines: collapse all internal whitespace runs to one
#             space, strip leading/trailing.
#             Drop blank lines from both categories.
#
# stdin:  raw HTML (UTF-8)
# stdout: clean signal text; code blocks whitespace-preserved
# ─────────────────────────────────────────────────────────────────────────────

# ── Pass 1: erase single-line HTML comments ──────────────────────────────────
sed 's/<!--[^-]*-->//g' |

# ── Pass 2: zone + code-block extraction ─────────────────────────────────────
awk '
BEGIN {
    m          = 0   # inside <main> zone
    s          = 0   # inside noise sub-zone
    pre        = 0   # inside <pre> block (whitespace-significant)
    in_comment = 0   # inside multi-line HTML comment
    SENT       = "\002"   # STX sentinel — marks code lines
}

# ── Multi-line comment handling ──────────────────────────────────────────────
/<!--/ && !/-->/ {
    in_comment = 1
    sub(/<!--.*/, "")
    if (m && !s && length($0)) print (pre ? SENT : "") $0
    next
}
in_comment && /-->/ {
    sub(/.*-->/, "")
    in_comment = 0
    if (m && !s && length($0)) print (pre ? SENT : "") $0
    next
}
in_comment { next }

# ── Main zone entry ──────────────────────────────────────────────────────────
!m && /<main[^>]*>/ {
    m = 1
    sub(/.*<main[^>]*>/, "")
    # fall through — same line may have content
}

# ── Main zone exit ───────────────────────────────────────────────────────────
m && /<\/main>/ {
    sub(/<\/main>.*/, "")
    if (!s && length($0)) print (pre ? SENT : "") $0
    m = 0
    next
}

# ── Noise sub-zone suppression ───────────────────────────────────────────────
# Class-based sidebar detection: elements with sidebar/toc/menu/nav in class
m && !s && !pre && /class="[^"]*\(sidebar\|toc\|menu-list\|page-nav\)[^"]*"/ {
    s = 1
    next
}
# Standard noise zones
m && !s && !pre && /<(nav|aside|footer|header|script|style|noscript)[[:space:]>]/ {
    s = 1
    next
}
m && s && /<\/(nav|aside|footer|header|script|style|noscript)>/ {
    s = 0
    next
}

# ── Code block tracking (enter) ──────────────────────────────────────────────
# <pre> opens a whitespace-significant block. The pre tag itself is noise
# (structural), but everything between <pre> and </pre> is signal.
# Entering pre: the opening tag line may have content after the tag.
m && !s && !pre && /<pre[^>]*>/ {
    pre = 1
    sub(/.*<pre[^>]*>/, "")       # strip the structural open tag
    # Check for same-line close (uncommon but valid: <pre>code</pre>)
    if (/<\/pre>/) {
        sub(/<\/pre>.*/, "")
        pre = 0
        if (length($0)) print SENT $0
        next
    }
    if (length($0)) print SENT $0
    next
}

# ── Code block tracking (exit) ───────────────────────────────────────────────
m && !s && pre && /<\/pre>/ {
    sub(/<\/pre>.*/, "")          # content before </pre> is still signal
    if (length($0)) print SENT $0
    pre = 0
    next
}

# ── Signal output ────────────────────────────────────────────────────────────
m && !s { print (pre ? SENT : "") $0 }
' |

# ── Pass 3: tag removal ───────────────────────────────────────────────────────
# Remove all HTML tags uniformly. The STX sentinel is not affected by this
# because it precedes the tag content, not inside it.
# Entity decode applied here for both prose and code lines.
sed '
s/<[^>]*>//g
s/&amp;/\&/g
s/&lt;/</g
s/&gt;/>/g
s/&nbsp;/ /g
s/&quot;/"/g
s/&#39;/'"'"'/g
' |

# ── Pass 4: differential whitespace handling ──────────────────────────────────
awk '
BEGIN { SENT = "\002" }
{
    # Detect code line
    if (substr($0, 1, 1) == SENT) {
        line = substr($0, 2)           # strip sentinel
        # Drop blank code lines (empty pre lines add no value)
        if (line ~ /^[[:space:]]*$/) next
        # Strip trailing whitespace only; preserve leading (indentation = syntax)
        gsub(/[[:space:]]+$/, "", line)
        print line
    } else {
        # Prose line: collapse all whitespace, drop if blank
        gsub(/[[:space:]]+/, " ", $0)
        gsub(/^ | $/, "", $0)
        if (length($0)) print
    }
}
'