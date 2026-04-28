#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# recipes/hardcoded/generic_html.sh
# Topology class: GENERIC_HTML (universal fallback)
#
# Conservative global noise stripper. Used when no specific recipe is
# registered for the incoming topology class.
#
# Signal zone: everything NOT in the noise containers listed below.
# Noise stripped: <nav> <header> <footer> <aside> <script> <style>
#                 <noscript> <form> <iframe> <svg> <template>
#                 HTML comments (both single-line and multi-line).
#
# Architecture: three-pass pipeline, all streaming, no temp files.
#
# This recipe WILL retain some noise — breadcrumb divs, cookie banners,
# and custom layouts that do not use standard HTML5 sectioning elements
# will pass through.  That is acceptable: the goal is noise REDUCTION,
# not noise ELIMINATION.  On known topologies the specific recipes achieve
# 60-80% reduction.  GENERIC_HTML achieves 30-50%.  Still better than raw.
#
# stdin:  raw HTML (UTF-8)
# stdout: de-noised text, one logical sentence or heading per line
# ─────────────────────────────────────────────────────────────────────────────

# ── Pass 1: erase single-line HTML comments ──────────────────────────────────
sed 's/<!--[^-]*-->//g' |

# ── Pass 2: global noise zone state machine ──────────────────────────────────
awk '
BEGIN {
    s          = 0   # inside noise sub-zone (0/1)
    in_comment = 0   # inside multi-line HTML comment (0/1)
}

# ── Multi-line comment handling ──────────────────────────────────────────────
/<!--/ && !/>/ {
    in_comment = 1
    sub(/<!--.*/, "")
    if (!s && length($0)) print
    next
}
in_comment && /-->/ {
    sub(/.*-->/, "")
    in_comment = 0
    if (!s && length($0)) print
    next
}
in_comment { next }

# ── Noise zone entry ─────────────────────────────────────────────────────────
!s && /<(nav|header|footer|aside|script|style|noscript|form|iframe|svg|template)[[:space:]>]/ {
    s = 1
    next
}

# ── Noise zone exit ──────────────────────────────────────────────────────────
s && /<\/(nav|header|footer|aside|script|style|noscript|form|iframe|svg|template)>/ {
    s = 0
    next
}

# ── Discard inside noise zone ────────────────────────────────────────────────
s { next }

# ── Signal output ────────────────────────────────────────────────────────────
length($0) { print }
' |

# ── Pass 3: tag removal, entity decode, whitespace collapse ──────────────────
sed '
s/<[^>]*>//g
s/&amp;/\&/g
s/&lt;/</g
s/&gt;/>/g
s/&nbsp;/ /g
s/&quot;/"/g
s/&#39;/'"'"'/g
s/&#8216;/'"'"'/g
s/&#8217;/'"'"'/g
s/&#8220;/"/g
s/&#8221;/"/g
s/&#8212;/—/g
s/&#8211;/–/g
' |

grep -v '^[[:space:]]*$' |

sed 's/[[:space:]]\{1,\}/ /g
     s/^ //
     s/ $//'
