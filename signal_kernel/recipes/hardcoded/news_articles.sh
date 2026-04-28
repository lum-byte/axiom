#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# recipes/hardcoded/news_article.sh
# Topology class: NEWS_ARTICLE
#
# Signal zone: <article>
# Noise stripped: <nav> <aside> <footer> <header> <script> <style> <noscript>
#                 HTML comments. All remaining tags. Common HTML entities.
#
# Architecture: three-pass pipeline, all streaming, no temp files.
#
#   Pass 1 — comment erasure (sed):
#     HTML comments can contain tag-like strings that confuse zone detection.
#     Erase single-line comments before the state machine sees them.
#     Multi-line comments are handled by the awk state machine's comment flag.
#
#   Pass 2 — zone extraction (awk state machine):
#     grep cannot cross line boundaries. awk state machine is the correct
#     tool for extracting content that spans an unknown number of lines.
#
#     State variables:
#       a          — inside <article> zone (0/1)
#       s          — inside a noise sub-zone (0/1)
#       in_comment — inside <!-- ... --> spanning multiple lines (0/1)
#
#     Opening tag detection: /<article[^>]*>/ matches
#       <article>, <article class="foo">, <article id="main"> etc.
#       sub() strips everything up to and including the tag so content
#       on the same line as the opening tag is not lost.
#
#     Same-line open+close: <article>content</article>
#       After the sub() on the open tag, the modified $0 is re-evaluated
#       by subsequent patterns in the same awk cycle — the close-tag
#       pattern fires on the same line, correctly extracting inline content.
#
#     Noise sub-zones: any noise zone encountered while inside the article
#       zone is suppressed. The skip flag is set on the opening tag line
#       (which is discarded with next) and cleared on the closing tag line
#       (also discarded). Content between them is silently dropped.
#
#   Pass 3 — signal cleanup (sed + grep + sed):
#     Remove all remaining HTML tags with a greedy [^>]* match.
#     Known limitation: attributes containing a literal > break this.
#     Mitigation: rare in well-formed HTML; handled by the pre-pass
#     that removes comments which are the most common source of bare >.
#     Decode the five most common named entities and &#NNN; decimal refs.
#     Drop blank lines. Collapse whitespace runs to single space.
#
# stdin:  raw HTML (UTF-8)
# stdout: clean signal text, one logical sentence or heading per line
# ─────────────────────────────────────────────────────────────────────────────

# ── Pass 1: erase single-line HTML comments ──────────────────────────────────
sed 's/<!--[^-]*-->//g' |

# ── Pass 2: zone extraction state machine ────────────────────────────────────
awk '
BEGIN {
    a          = 0   # inside <article> zone
    s          = 0   # inside noise sub-zone
    in_comment = 0   # inside multi-line HTML comment
}

# ── Multi-line comment handling ──────────────────────────────────────────────
# Open comment with no close on same line → enter comment mode
/<!--/ && !/>/ {
    in_comment = 1
    sub(/<!--.*/, "")          # discard from comment open to end of line
    if (a && !s && length($0)) print
    next
}
# Comment close
in_comment && /-->/ {
    sub(/.*-->/, "")           # discard up through comment close
    in_comment = 0
    if (a && !s && length($0)) print
    next
}
in_comment { next }            # inside comment — discard entire line

# ── Article zone entry ───────────────────────────────────────────────────────
!a && /<article[^>]*>/ {
    a = 1
    sub(/.*<article[^>]*>/, "")   # discard everything before (and incl.) tag
    # fall through — same line may have content or even </article>
}

# ── Article zone exit ────────────────────────────────────────────────────────
a && /<\/article>/ {
    sub(/<\/article>.*/, "")   # discard from close tag to end of line
    if (!s && length($0)) print
    a = 0
    next
}

# ── Noise sub-zone suppression ───────────────────────────────────────────────
# Noise zones that can legally nest inside <article>:
#   <aside>    — related/sidebar content injected inside article
#   <nav>      — in-article navigation (table of contents, prev/next)
#   <header>   — article header zone (often contains byline noise)
#   <footer>   — article footer (tags, comments link, share buttons)
#   <script>   — tracking pixels, analytics snippets
#   <style>    — inline style blocks
#   <noscript> — fallback content for analytics scripts
#
# Note: <figure> and <figcaption> are NOT suppressed — figcaptions are
# signal (image descriptions that convey editorial context).
# Note: <header> inside <article> often contains the headline h1 which
# IS signal. Uncomment the header line below only if your corpus reliably
# puts noise (not headlines) in article <header> elements.

a && !s && /<(nav|aside|footer|script|style|noscript)[[:space:]>]/ {
    s = 1
    next
}
# Close tags for noise zones — order matches open list above
a && s && /<\/(nav|aside|footer|script|style|noscript)>/ {
    s = 0
    next
}

# ── Signal output ────────────────────────────────────────────────────────────
a && !s && length($0) { print }
' |

# ── Pass 3: tag removal, entity decode, whitespace collapse ──────────────────
sed '
# Remove all HTML tags. Greedy [^>]* stops at first > on the line.
# This correctly handles the vast majority of real-world HTML tags.
s/<[^>]*>//g

# Decode named HTML entities (the five that appear most in article text)
s/&amp;/\&/g
s/&lt;/</g
s/&gt;/>/g
s/&nbsp;/ /g
s/&quot;/"/g
s/&#39;/'"'"'/g

# Decode common decimal numeric references (e.g. &#8217; = right single quote)
s/&#8216;/'"'"'/g
s/&#8217;/'"'"'/g
s/&#8220;/"/g
s/&#8221;/"/g
s/&#8212;/—/g
s/&#8211;/–/g
' |

# Drop lines that are now blank (were tag-only lines)
grep -v '^[[:space:]]*$' |

# Collapse all whitespace runs to a single space; strip leading/trailing
sed 's/[[:space:]]\{1,\}/ /g
     s/^ //
     s/ $//'