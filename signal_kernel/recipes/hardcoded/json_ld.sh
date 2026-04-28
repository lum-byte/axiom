#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# recipes/hardcoded/json_ld.sh
# Topology class: JSON_LD_STRUCTURED
#
# Signal: <script type="application/ld+json"> blocks in <head> only.
# Noise:  Everything outside <head>. JSON-LD in <body> is CMS-injected
#         widget noise (WebSite, SearchAction, BreadcrumbList appended by
#         third-party plugins). Only <head> JSON-LD reflects editorial intent.
#
# Architecture: two-pass awk pipeline.
#
#   Pass 1 — HEAD isolation:
#     Discards everything from </head> onward. awk exits immediately on
#     </head> — it does not read or process the body. This is critical for
#     large pages where the body may be hundreds of kilobytes of noise.
#     The kernel should not spend time reading content it will discard.
#     awk `exit` achieves this; grep with `-m` count would cut lines but
#     not terminate the process reading stdin, causing pipe stalls on
#     large inputs. awk exit properly closes stdin.
#
#   Pass 2 — JSON-LD script block extraction:
#     Scans the head section for <script type="application/ld+json"> blocks.
#     Handles both attribute quote styles (" and ').
#     Handles whitespace variations in the type attribute.
#     Extracts content between the script tags. Discards the tags themselves.
#     Emits each JSON-LD block as a separate output unit separated by a blank
#     line — multiple blocks in a single page head are each emitted as a unit.
#
#     The `in_script` state is set on the opening tag line and cleared on the
#     closing </script> tag line. Content between these is emitted verbatim
#     (no whitespace modification — JSON-LD is structured data, not prose).
#
#   Important: type attribute matching is intentionally broad.
#     Matches: type="application/ld+json"
#     Matches: type='application/ld+json'
#     Matches: type = "application/ld+json"  (with spaces, though unusual)
#     Does NOT match: type="application/json" (different MIME type — not LD)
#
# stdin:  raw HTML (UTF-8) — full page
# stdout: JSON-LD object(s) from <head>, one per block, separated by blank line
# ─────────────────────────────────────────────────────────────────────────────

# ── Pass 1: HEAD isolation — exit immediately at </head> ─────────────────────
# awk exit terminates stdin reading: no pipe stall on large bodies.
# Case-insensitive: handles </HEAD>, </Head> from legacy HTML generators.
awk '
tolower($0) ~ /<\/head>/ { exit }
{ print }
' |

# ── Pass 2: JSON-LD block extraction ─────────────────────────────────────────
awk '
BEGIN {
    in_script  = 0   # currently inside a ld+json script block
    block_count = 0  # number of blocks emitted (for separator blank line)
}

# ── Script block entry ────────────────────────────────────────────────────────
# Matches the opening script tag with ld+json type attribute.
# Pattern covers:
#   <script type="application/ld+json">
#   <script type='"'"'application/ld+json'"'"'>
#   <SCRIPT TYPE="APPLICATION/LD+JSON">  (uppercased by some CMSes)
# We use tolower() for case-insensitive matching without requiring grep -i.
!in_script && tolower($0) ~ /<script[^>]*type=[^>]*application\/ld\+json/ {
    in_script = 1
    # The opening tag may have content on the same line after the >
    # Strip everything up to and including the closing > of the script tag
    line = $0
    if (sub(/.*<[Ss][Cc][Rr][Ii][Pp][Tt][^>]*>/, "", line) && length(line)) {
        # Check for same-line close (unusual but valid)
        if (line ~ /<\/[Ss][Cc][Rr][Ii][Pp][Tt]>/) {
            sub(/<\/[Ss][Cc][Rr][Ii][Pp][Tt]>.*/, "", line)
            if (length(line)) {
                if (block_count > 0) print ""
                print line
                block_count++
            }
            in_script = 0
        } else {
            if (block_count > 0) print ""
            block_count++
            if (length(line)) print line
        }
    } else {
        # Tag opened but no inline content on this line
        if (block_count > 0) print ""
        block_count++
    }
    next
}

# ── Script block exit ─────────────────────────────────────────────────────────
in_script && tolower($0) ~ /<\/script>/ {
    # Capture any content before the closing tag on this line
    line = $0
    sub(/<\/[Ss][Cc][Rr][Ii][Pp][Tt]>.*/, "", line)
    if (length(line) && line !~ /^[[:space:]]*$/) print line
    in_script = 0
    next
}

# ── Content output ────────────────────────────────────────────────────────────
# Emit lines inside script blocks verbatim — JSON-LD is structured data.
# Blank lines inside a JSON-LD block are retained (may be pretty-printing).
in_script { print }
'