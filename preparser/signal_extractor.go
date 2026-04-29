package preparser

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"sync"
	"regexp"
)

type ZoneType string

const (
	ZoneProse      ZoneType = "prose"
	ZoneCode       ZoneType = "code"
	ZoneTable      ZoneType = "table"
	ZoneList       ZoneType = "list"
	ZoneHeading    ZoneType = "heading"
	ZoneMetadata   ZoneType = "metadata"
	ZoneNavigation ZoneType = "navigation"
	ZoneQuote      ZoneType = "quote"
	ZoneUnknown    ZoneType = "unknown"
)

const (
	DefaultMinZoneTokens = 3
	DefaultMaxZoneBytes  = 64 * 1024
)

type SignalExtractedEvent struct {
	URL             string  `json:"url"`
	TopologyClass   string  `json:"topology_class"`
	SignalType      string  `json:"signal_type"`
	ByteCount       int     `json:"byte_count"`
	TokenCount      int     `json:"token_count"`
	SignalDensity   float64 `json:"signal_density"`
	ZoneCount       int     `json:"zone_count"`
	SourceComponent string  `json:"source_component"`
	RunID           string  `json:"run_id"`
}

type SignalExtractionInput struct {
	URL           string
	TopologyClass string
	SanitizedText string
	RunID         string
}

type ZoneExtractionOptions struct {
	MinTokens          int      `json:"min_tokens"`
	MaxZoneBytes       int      `json:"max_zone_bytes"`
	KeepNavigation     bool     `json:"keep_navigation"`
	PreferCodeLanguage bool     `json:"prefer_code_language"`
	AllowedTypes       []ZoneType `json:"allowed_types"`
	NowUnix            int64    `json:"now_unix"`
}

type SignalZone struct {
	ID              string            `json:"id"`
	Type            ZoneType          `json:"type"`
	Content         string            `json:"content"`
	StartByte       int               `json:"start_byte"`
	EndByte         int               `json:"end_byte"`
	TokenCount      int               `json:"token_count"`
	ByteCount       int               `json:"byte_count"`
	Density         float64           `json:"density"`
	Confidence      float64           `json:"confidence"`
	Language        string            `json:"language"`
	TopologyClass   string            `json:"topology_class"`
	Attributes      map[string]string `json:"attributes"`
	Rank            int               `json:"rank"`
	ReductionWeight float64           `json:"reduction_weight"`
}

type ExtractedSignal struct {
	URL              string       `json:"url"`
	Domain           string       `json:"domain"`
	TopologyClass    string       `json:"topology_class"`
	Zones            []SignalZone `json:"zones"`
	TotalSignalBytes int          `json:"total_signal_bytes"`
	TotalTokens      int          `json:"total_tokens"`
	ReductionRatio   float64      `json:"reduction_ratio"`
	SignalDensity    float64      `json:"signal_density"`
	ExtractedAtUnix  int64        `json:"extracted_at_unix"`
	RunID            string       `json:"run_id"`
}

type ZoneCandidate struct {
	Content   string
	StartByte int
	EndByte   int
	TypeHint  ZoneType
}

func ExtractSignal(input SignalExtractionInput) (SignalExtractedEvent, error) {
	if input.URL == "" {
		return SignalExtractedEvent{}, errors.New("url is empty")
	}
	if input.TopologyClass == "" {
		return SignalExtractedEvent{}, errors.New("topology_class is empty")
	}
	if input.RunID == "" {
		return SignalExtractedEvent{}, errors.New("run_id is empty")
	}
	text := strings.TrimSpace(input.SanitizedText)
	extracted, err := ExtractStructuredSignal(input, ZoneExtractionOptions{})
	if err != nil {
		return SignalExtractedEvent{}, err
	}
	tokens := extracted.TotalTokens
	zones := len(extracted.Zones)
	event := SignalExtractedEvent{
		URL:             input.URL,
		TopologyClass:   input.TopologyClass,
		SignalType:      classifySignalFromZones(text, extracted.Zones),
		ByteCount:       len([]byte(text)),
		TokenCount:      tokens,
		SignalDensity:   extracted.SignalDensity,
		ZoneCount:       zones,
		SourceComponent: "preparser.signal_extractor",
		RunID:           input.RunID,
	}
	return event, nil
}

func ExtractStructuredSignal(input SignalExtractionInput, opts ZoneExtractionOptions) (ExtractedSignal, error) {
	if input.URL == "" {
		return ExtractedSignal{}, errors.New("url is empty")
	}
	if input.TopologyClass == "" {
		return ExtractedSignal{}, errors.New("topology_class is empty")
	}
	if input.RunID == "" {
		return ExtractedSignal{}, errors.New("run_id is empty")
	}
	if opts.MinTokens <= 0 {
		opts.MinTokens = DefaultMinZoneTokens
	}
	if opts.MaxZoneBytes <= 0 {
		opts.MaxZoneBytes = DefaultMaxZoneBytes
	}
	if opts.NowUnix <= 0 {
		opts.NowUnix = time.Now().Unix()
	}
	text := strings.TrimSpace(input.SanitizedText)
	candidates := SplitSignalCandidates(text)
	zones := make([]SignalZone, 0, len(candidates))
	allowed := allowedZoneSet(opts.AllowedTypes)
	for _, candidate := range candidates {
		zone := BuildSignalZone(input.URL, input.TopologyClass, candidate, opts)
		if zone.TokenCount < opts.MinTokens {
			continue
		}
		if zone.ByteCount > opts.MaxZoneBytes {
			for _, chunk := range splitLargeZone(zone, opts.MaxZoneBytes) {
				if zoneAllowed(chunk, allowed, opts.KeepNavigation) {
					zones = append(zones, chunk)
				}
			}
			continue
		}
		if zoneAllowed(zone, allowed, opts.KeepNavigation) {
			zones = append(zones, zone)
		}
	}
	RankSignalZones(zones)
	totalBytes := 0
	totalTokens := 0
	for _, zone := range zones {
		totalBytes += zone.ByteCount
		totalTokens += zone.TokenCount
	}
	rawBytes := len([]byte(text))
	ratio := 0.0
	if rawBytes > 0 {
		ratio = float64(totalBytes) / float64(rawBytes)
	}
	return ExtractedSignal{
		URL:              input.URL,
		Domain:           domainFromURL(input.URL),
		TopologyClass:    input.TopologyClass,
		Zones:            zones,
		TotalSignalBytes: totalBytes,
		TotalTokens:      totalTokens,
		ReductionRatio:   ratio,
		SignalDensity:    weightedZoneDensity(zones, rawBytes),
		ExtractedAtUnix:  opts.NowUnix,
		RunID:            input.RunID,
	}, nil
}

func (e SignalExtractedEvent) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "signal_extracted", Component: "preparser.signal_extractor", Payload: e}
}

func (s ExtractedSignal) Event() SignalExtractedEvent {
	return SignalExtractedEvent{
		URL:             s.URL,
		TopologyClass:   s.TopologyClass,
		SignalType:      dominantZoneType(s.Zones),
		ByteCount:       s.TotalSignalBytes,
		TokenCount:      s.TotalTokens,
		SignalDensity:   s.SignalDensity,
		ZoneCount:       len(s.Zones),
		SourceComponent: "preparser.signal_extractor",
		RunID:           s.RunID,
	}
}

func (s ExtractedSignal) BridgeEvent() BridgeRequest {
	return s.Event().BridgeEvent()
}

func SplitSignalCandidates(text string) []ZoneCandidate {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	blocks := splitByBlankLines(text)
	candidates := make([]ZoneCandidate, 0, len(blocks))
	offset := 0
	for _, block := range blocks {
		start := strings.Index(text[offset:], block)
		if start < 0 {
			start = offset
		} else {
			start += offset
		}
		end := start + len(block)
		offset = end
		for _, sub := range splitStructuredBlock(block, start) {
			candidates = append(candidates, sub)
		}
	}
	return mergeAdjacentCandidates(candidates)
}

func BuildSignalZone(sourceURL string, topologyClass string, candidate ZoneCandidate, opts ZoneExtractionOptions) SignalZone {
	content := strings.TrimSpace(candidate.Content)
	zoneType := candidate.TypeHint
	if zoneType == "" || zoneType == ZoneUnknown {
		zoneType = detectZoneType(content)
	}
	tokens := countTokens(content)
	byteCount := len([]byte(content))
	d := density(content)
	lang := ""
	if zoneType == ZoneCode || opts.PreferCodeLanguage {
		lang = detectCodeLanguage(content)
	}
	attrs := zoneAttributes(content, zoneType)
	conf := zoneConfidence(zoneType, tokens, d, attrs)
	return SignalZone{
		ID:              stableZoneID(sourceURL, candidate.StartByte, candidate.EndByte, content),
		Type:            zoneType,
		Content:         content,
		StartByte:       candidate.StartByte,
		EndByte:         candidate.EndByte,
		TokenCount:      tokens,
		ByteCount:       byteCount,
		Density:         d,
		Confidence:      conf,
		Language:        lang,
		TopologyClass:   topologyClass,
		Attributes:      attrs,
		ReductionWeight: reductionWeight(zoneType, topologyClass),
	}
}

func splitByBlankLines(text string) []string {
	var out []string
	var builder strings.Builder
	lines := strings.Split(text, "\n")
	blankRun := 0
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			blankRun++
			if blankRun >= 1 && strings.TrimSpace(builder.String()) != "" {
				out = append(out, strings.TrimSpace(builder.String()))
				builder.Reset()
			}
			continue
		}
		blankRun = 0
		if builder.Len() > 0 {
			builder.WriteByte('\n')
		}
		builder.WriteString(line)
	}
	if strings.TrimSpace(builder.String()) != "" {
		out = append(out, strings.TrimSpace(builder.String()))
	}
	if len(out) == 0 {
		return []string{text}
	}
	return out
}

func splitStructuredBlock(block string, baseOffset int) []ZoneCandidate {
	lines := strings.Split(block, "\n")
	if len(lines) <= 1 {
		return []ZoneCandidate{{Content: block, StartByte: baseOffset, EndByte: baseOffset + len(block), TypeHint: detectZoneType(block)}}
	}
	if looksLikeTable(block) || looksLikeCodeBlock(block) {
		return []ZoneCandidate{{Content: block, StartByte: baseOffset, EndByte: baseOffset + len(block), TypeHint: detectZoneType(block)}}
	}
	out := make([]ZoneCandidate, 0)
	var builder strings.Builder
	sectionStart := baseOffset
	offset := baseOffset
	currentType := ZoneUnknown
	flush := func(end int) {
		content := strings.TrimSpace(builder.String())
		if content != "" {
			out = append(out, ZoneCandidate{Content: content, StartByte: sectionStart, EndByte: end, TypeHint: currentType})
		}
		builder.Reset()
	}
	for _, line := range lines {
		lineType := detectLineZoneType(line)
		if currentType != ZoneUnknown && lineType != currentType && strings.TrimSpace(builder.String()) != "" {
			flush(offset)
			sectionStart = offset
		}
		if builder.Len() > 0 {
			builder.WriteByte('\n')
		}
		builder.WriteString(line)
		currentType = lineType
		offset += len(line) + 1
	}
	flush(baseOffset + len(block))
	if len(out) == 0 {
		out = append(out, ZoneCandidate{Content: block, StartByte: baseOffset, EndByte: baseOffset + len(block), TypeHint: detectZoneType(block)})
	}
	return out
}

func mergeAdjacentCandidates(in []ZoneCandidate) []ZoneCandidate {
	if len(in) <= 1 {
		return in
	}
	out := make([]ZoneCandidate, 0, len(in))
	for _, candidate := range in {
		if strings.TrimSpace(candidate.Content) == "" {
			continue
		}
		if len(out) == 0 {
			out = append(out, candidate)
			continue
		}
		last := &out[len(out)-1]
		if last.TypeHint == candidate.TypeHint && len(last.Content)+len(candidate.Content) < DefaultMaxZoneBytes/2 {
			last.Content = strings.TrimSpace(last.Content + "\n" + candidate.Content)
			last.EndByte = candidate.EndByte
			continue
		}
		out = append(out, candidate)
	}
	return out
}

func countTokens(text string) int {
	inToken := false
	count := 0
	for _, r := range text {
		if unicode.IsSpace(r) {
			inToken = false
			continue
		}
		if !inToken {
			count++
			inToken = true
		}
	}
	return count
}

func countZones(text string) int {
	if text == "" {
		return 0
	}
	zones := 1
	for _, sep := range []string{"\n\n", "<section", "<article", "```"} {
		zones += strings.Count(strings.ToLower(text), sep)
	}
	return zones
}

func density(text string) float64 {
	if text == "" {
		return 0
	}
	signal := 0
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			signal++
		}
	}
	return float64(signal) / float64(len([]rune(text)))
}

func classifySignal(text string) string {
	lower := strings.ToLower(text)
	codeHits := 0
	for _, kw := range []string{"func ", "class ", "def ", "return ", "import ", "const ", "let ", "var ", "SELECT ", "curl "} {
		if strings.Contains(text, kw) || strings.Contains(lower, strings.ToLower(kw)) {
			codeHits++
		}
	}
	switch {
	case strings.Contains(lower, "<table") || strings.Contains(lower, "|---"):
		return "table"
	case codeHits >= 2:
		return "code"
	case strings.Contains(lower, "<h1") || strings.HasPrefix(strings.TrimSpace(text), "#"):
		return "heading"
	case strings.Contains(lower, "<li") || strings.Contains(text, "\n- "):
		return "list"
	default:
		return "prose"
	}
}

func classifySignalFromZones(text string, zones []SignalZone) string {
	if len(zones) == 0 {
		return classifySignal(text)
	}
	return string(dominantZoneType(zones))
}

func dominantZoneType(zones []SignalZone) string {
	if len(zones) == 0 {
		return string(ZoneUnknown)
	}
	counts := make(map[ZoneType]float64)
	for _, zone := range zones {
		counts[zone.Type] += float64(zone.TokenCount) * clampFloat(zone.Confidence, 0.1, 1)
	}
	best := ZoneUnknown
	bestScore := -1.0
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, string(key))
	}
	sort.Strings(keys)
	for _, key := range keys {
		zt := ZoneType(key)
		score := counts[zt]
		if score > bestScore {
			best = zt
			bestScore = score
		}
	}
	return string(best)
}

func detectZoneType(content string) ZoneType {
	trimmed := strings.TrimSpace(content)
	lower := strings.ToLower(trimmed)
	switch {
	case trimmed == "":
		return ZoneUnknown
	case looksLikeTable(trimmed):
		return ZoneTable
	case looksLikeCodeBlock(trimmed):
		return ZoneCode
	case strings.HasPrefix(trimmed, "#") || strings.HasPrefix(lower, "<h1") || strings.HasPrefix(lower, "<h2") || isShortTitle(trimmed):
		return ZoneHeading
	case looksLikeList(trimmed):
		return ZoneList
	case strings.HasPrefix(trimmed, ">") || strings.HasPrefix(lower, "<blockquote"):
		return ZoneQuote
	case looksLikeNavigation(trimmed):
		return ZoneNavigation
	case looksLikeMetadata(trimmed):
		return ZoneMetadata
	default:
		return ZoneProse
	}
}

func detectLineZoneType(line string) ZoneType {
	line = strings.TrimSpace(line)
	if line == "" {
		return ZoneUnknown
	}
	return detectZoneType(line)
}

func looksLikeTable(text string) bool {
	lower := strings.ToLower(text)
	if strings.Contains(lower, "<table") || strings.Contains(lower, "</tr>") {
		return true
	}
	lines := strings.Split(text, "\n")
	pipeRows := 0
	for _, line := range lines {
		if strings.Count(line, "|") >= 2 {
			pipeRows++
		}
	}
	return pipeRows >= 2 || strings.Contains(text, "|---")
}

func looksLikeCodeBlock(text string) bool {
	lower := strings.ToLower(text)
	if strings.Contains(text, "```") || strings.Contains(lower, "<pre") || strings.Contains(lower, "<code") {
		return true
	}
	keywords := []string{"func ", "class ", "def ", "return ", "import ", "const ", "let ", "var ", "SELECT ", "curl ", "package ", "#include", "fn "}
	hits := 0
	for _, kw := range keywords {
		if strings.Contains(text, kw) || strings.Contains(lower, strings.ToLower(kw)) {
			hits++
		}
	}
	if hits >= 2 {
		return true
	}
	lines := strings.Split(text, "\n")
	indented := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "    ") || strings.HasPrefix(line, "\t") {
			indented++
		}
	}
	return len(lines) >= 3 && indented*2 >= len(lines)
}

func looksLikeList(text string) bool {
	lines := strings.Split(text, "\n")
	if len(lines) < 2 {
		return strings.HasPrefix(strings.TrimSpace(text), "<li")
	}
	items := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") || strings.HasPrefix(line, "<li") || numberedListLine(line) {
			items++
		}
	}
	return items >= 2 && items*2 >= len(lines)
}

func numberedListLine(line string) bool {
	if line == "" {
		return false
	}
	i := 0
	for i < len(line) && line[i] >= '0' && line[i] <= '9' {
		i++
	}
	return i > 0 && i+1 < len(line) && (line[i] == '.' || line[i] == ')') && line[i+1] == ' '
}

func looksLikeNavigation(text string) bool {
	lower := strings.ToLower(text)
	navWords := []string{"home", "login", "sign in", "privacy", "terms", "subscribe", "menu", "breadcrumb"}
	hits := 0
	for _, word := range navWords {
		if strings.Contains(lower, word) {
			hits++
		}
	}
	return hits >= 3 && countTokens(text) < 80
}

func looksLikeMetadata(text string) bool {
	lower := strings.ToLower(text)
	if strings.HasPrefix(lower, "published:") || strings.HasPrefix(lower, "author:") || strings.HasPrefix(lower, "updated:") {
		return true
	}
	colons := strings.Count(text, ":")
	return colons >= 2 && countTokens(text) < 40
}

func isShortTitle(text string) bool {
	tokens := countTokens(text)
	if tokens == 0 || tokens > 12 {
		return false
	}
	if strings.Contains(text, ".") || strings.Contains(text, ",") {
		return false
	}
	first := []rune(strings.TrimSpace(text))[0]
	return unicode.IsUpper(first)
}

func detectCodeLanguage(text string) string {
	lower := strings.ToLower(text)
	scores := map[string]int{
		"go":         keywordScore(lower, []string{"func ", "package ", "defer ", "chan ", "interface{"}),
		"python":     keywordScore(lower, []string{"def ", "import ", "async def ", "self.", "pytest"}),
		"javascript": keywordScore(lower, []string{"const ", "let ", "=>", "function ", "promise"}),
		"rust":       keywordScore(lower, []string{"fn ", "let mut", "impl ", "cargo", "pub struct"}),
		"c":          keywordScore(lower, []string{"#include", "malloc", "free(", "sizeof", "uint32_t"}),
		"sql":        keywordScore(lower, []string{"select ", "from ", "where ", "join ", "insert "}),
		"shell":      keywordScore(lower, []string{"#!/", "curl ", "grep ", "awk ", "export "}),
	}
	best := ""
	bestScore := 0
	keys := make([]string, 0, len(scores))
	for key := range scores {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if scores[key] > bestScore {
			best = key
			bestScore = scores[key]
		}
	}
	if bestScore == 0 {
		return ""
	}
	return best
}

func keywordScore(text string, keywords []string) int {
	score := 0
	for _, kw := range keywords {
		if strings.Contains(text, kw) {
			score++
		}
	}
	return score
}

func zoneAttributes(content string, zoneType ZoneType) map[string]string {
	attrs := make(map[string]string)
	lines := strings.Split(content, "\n")
	attrs["lines"] = strconv.Itoa(len(lines))
	attrs["characters"] = strconv.Itoa(len([]rune(content)))
	switch zoneType {
	case ZoneTable:
		attrs["rows"] = strconv.Itoa(countTableRows(content))
		attrs["columns"] = strconv.Itoa(estimateTableColumns(content))
	case ZoneCode:
		attrs["language"] = detectCodeLanguage(content)
	case ZoneList:
		attrs["items"] = strconv.Itoa(countListItems(content))
	case ZoneHeading:
		attrs["level"] = headingLevel(content)
	}
	return attrs
}

func countTableRows(content string) int {
	rows := 0
	for _, line := range strings.Split(content, "\n") {
		if strings.Count(line, "|") >= 2 || strings.Contains(strings.ToLower(line), "<tr") {
			rows++
		}
	}
	return rows
}

func estimateTableColumns(content string) int {
	best := 0
	for _, line := range strings.Split(content, "\n") {
		cols := strings.Count(line, "|") - 1
		if cols > best {
			best = cols
		}
	}
	if best < 0 {
		return 0
	}
	return best
}

func countListItems(content string) int {
	items := 0
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") || strings.HasPrefix(line, "<li") || numberedListLine(line) {
			items++
		}
	}
	return items
}

func headingLevel(content string) string {
	trimmed := strings.TrimSpace(content)
	for i := 1; i <= 6; i++ {
		if strings.HasPrefix(trimmed, strings.Repeat("#", i)+" ") {
			return strconv.Itoa(i)
		}
		if strings.HasPrefix(strings.ToLower(trimmed), "<h"+strconv.Itoa(i)) {
			return strconv.Itoa(i)
		}
	}
	return "1"
}

func zoneConfidence(zoneType ZoneType, tokens int, d float64, attrs map[string]string) float64 {
	base := 0.45
	switch zoneType {
	case ZoneCode, ZoneTable:
		base = 0.80
	case ZoneHeading:
		base = 0.70
	case ZoneList:
		base = 0.65
	case ZoneProse:
		base = 0.60
	case ZoneNavigation:
		base = 0.20
	case ZoneMetadata:
		base = 0.35
	}
	tokenScore := math.Min(1, math.Log(float64(tokens+1))/math.Log(200))
	conf := base*0.6 + tokenScore*0.25 + clampFloat(d, 0, 1)*0.15
	if attrs["language"] != "" {
		conf += 0.05
	}
	return clampFloat(conf, 0, 1)
}

func reductionWeight(zoneType ZoneType, topologyClass string) float64 {
	switch zoneType {
	case ZoneNavigation:
		return 0.05
	case ZoneMetadata:
		return 0.25
	case ZoneHeading:
		return 0.80
	case ZoneCode:
		if topologyClass == TopologySaaSDocs || topologyClass == TopologyRESTAPIJSON {
			return 1.0
		}
		return 0.85
	case ZoneTable:
		return 0.95
	case ZoneList:
		return 0.75
	default:
		return 0.65
	}
}

func RankSignalZones(zones []SignalZone) {
	sort.SliceStable(zones, func(i, j int) bool {
		left := zoneRankScore(zones[i])
		right := zoneRankScore(zones[j])
		if left != right {
			return left > right
		}
		return zones[i].StartByte < zones[j].StartByte
	})
	for i := range zones {
		zones[i].Rank = i + 1
	}
}

func zoneRankScore(zone SignalZone) float64 {
	lengthScore := math.Min(1, math.Log(float64(zone.TokenCount+1))/math.Log(500))
	return zone.Confidence*0.50 + zone.Density*0.20 + zone.ReductionWeight*0.20 + lengthScore*0.10
}

func weightedZoneDensity(zones []SignalZone, rawBytes int) float64 {
	if rawBytes <= 0 || len(zones) == 0 {
		return 0
	}
	weighted := 0.0
	for _, zone := range zones {
		weighted += float64(zone.ByteCount) * zone.Density * zone.ReductionWeight
	}
	return clampFloat(weighted/float64(rawBytes), 0, 1)
}

func allowedZoneSet(types []ZoneType) map[ZoneType]bool {
	if len(types) == 0 {
		return nil
	}
	out := make(map[ZoneType]bool, len(types))
	for _, zt := range types {
		out[zt] = true
	}
	return out
}

func zoneAllowed(zone SignalZone, allowed map[ZoneType]bool, keepNavigation bool) bool {
	if zone.Type == ZoneNavigation && !keepNavigation {
		return false
	}
	if allowed == nil {
		return true
	}
	return allowed[zone.Type]
}

func splitLargeZone(zone SignalZone, maxBytes int) []SignalZone {
	if maxBytes <= 0 || zone.ByteCount <= maxBytes {
		return []SignalZone{zone}
	}
	paragraphs := splitByBlankLines(zone.Content)
	out := make([]SignalZone, 0, len(paragraphs))
	offset := zone.StartByte
	for i, paragraph := range paragraphs {
		if len([]byte(paragraph)) > maxBytes {
			for _, chunk := range chunkStringByBytes(paragraph, maxBytes) {
				out = append(out, cloneZoneChunk(zone, chunk, offset, i))
				offset += len(chunk)
			}
			continue
		}
		out = append(out, cloneZoneChunk(zone, paragraph, offset, i))
		offset += len(paragraph)
	}
	return out
}

func chunkStringByBytes(text string, maxBytes int) []string {
	if len([]byte(text)) <= maxBytes {
		return []string{text}
	}
	var out []string
	var builder strings.Builder
	for _, r := range text {
		if builder.Len()+len(string(r)) > maxBytes && builder.Len() > 0 {
			out = append(out, builder.String())
			builder.Reset()
		}
		builder.WriteRune(r)
	}
	if builder.Len() > 0 {
		out = append(out, builder.String())
	}
	return out
}

func cloneZoneChunk(zone SignalZone, content string, start int, ordinal int) SignalZone {
	content = strings.TrimSpace(content)
	byteCount := len([]byte(content))
	zone.Content = content
	zone.StartByte = start
	zone.EndByte = start + byteCount
	zone.TokenCount = countTokens(content)
	zone.ByteCount = byteCount
	zone.Density = density(content)
	zone.ID = stableZoneID(zone.ID, start, start+byteCount, content+"#"+strconv.Itoa(ordinal))
	return zone
}

func stableZoneID(sourceURL string, start int, end int, content string) string {
	sum := sha256.Sum256([]byte(sourceURL + ":" + strconv.Itoa(start) + ":" + strconv.Itoa(end) + ":" + content))
	return hex.EncodeToString(sum[:16])
}

func domainFromURL(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return normalizeDomain(parsed.Host)
}

// ─── Extraction Pipeline Interface ──────────────────────────────────────────

// ExtractorPlugin defines a discrete step in the extraction pipeline.
type ExtractorPlugin interface {
	Name() string
	Process(zone *SignalZone) error
	IsApplicable(topologyClass string) bool
}

// ExtractionPipeline orchestrates a series of extractor plugins over a zone.
type ExtractionPipeline struct {
	plugins []ExtractorPlugin
}

// NewExtractionPipeline creates an empty pipeline.
func NewExtractionPipeline() *ExtractionPipeline {
	return &ExtractionPipeline{
		plugins: make([]ExtractorPlugin, 0, 8),
	}
}

// AddPlugin appends a step to the pipeline.
func (ep *ExtractionPipeline) AddPlugin(plugin ExtractorPlugin) {
	if ep == nil || plugin == nil {
		return
	}
	ep.plugins = append(ep.plugins, plugin)
}

// Run executes the pipeline sequentially over the provided zone.
func (ep *ExtractionPipeline) Run(zone *SignalZone, topologyClass string) error {
	if ep == nil || zone == nil {
		return errors.New("invalid pipeline or zone")
	}
	for _, p := range ep.plugins {
		if p.IsApplicable(topologyClass) {
			if err := p.Process(zone); err != nil {
				return err // Hard stop on plugin error
			}
		}
	}
	// Re-evaluate metrics after pipeline
	zone.ByteCount = len([]byte(zone.Content))
	zone.TokenCount = countTokens(zone.Content)
	zone.Density = density(zone.Content)
	return nil
}

// ─── Zone Merging Strategies ────────────────────────────────────────────────

// ZoneMerger combines contiguous or semantically related zones.
type ZoneMerger struct {
	MaxGapBytes    int
	MinDensity     float64
	MaxMergedBytes int
}

// DefaultZoneMerger returns sensible settings for merging adjacent content blocks.
func DefaultZoneMerger() ZoneMerger {
	return ZoneMerger{
		MaxGapBytes:    500, // Typical DOM gap between related paragraphs
		MinDensity:     0.4, // Requires reasonable text density to bridge
		MaxMergedBytes: 8192, // Keep merged zones under 8KB
	}
}

// Merge sorts and combines overlapping or adjacent zones.
func (m ZoneMerger) Merge(zones []SignalZone) []SignalZone {
	if len(zones) <= 1 {
		return zones
	}
	
	// Sort by start byte
	sorted := append([]SignalZone(nil), zones...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].StartByte < sorted[j].StartByte
	})

	var merged []SignalZone
	current := sorted[0]

	for i := 1; i < len(sorted); i++ {
		next := sorted[i]
		
		gap := next.StartByte - current.EndByte
		
		// Conditions to merge: overlapping OR (gap <= max and both densities are good)
		canMerge := gap <= 0 || (gap <= m.MaxGapBytes && current.Density >= m.MinDensity && next.Density >= m.MinDensity)
		wouldFit := (next.EndByte - current.StartByte) <= m.MaxMergedBytes
		
		if canMerge && wouldFit {
			// Combine content
			if gap > 0 {
				current.Content = current.Content + "\n\n" + next.Content
			} else {
				// Overlap - just take next's content if we assume monotonic advancing
				// In reality, robust DOM parsing prevents true overlap, but handle gracefully
				current.Content = current.Content + "\n" + next.Content
			}
			
			// Extend boundaries
			if next.EndByte > current.EndByte {
				current.EndByte = next.EndByte
			}
			
			// Recalculate metrics
			current.TokenCount += next.TokenCount
			current.ByteCount = len([]byte(current.Content))
			current.Density = density(current.Content)
			
			// Stable ID composition
			current.ID = stableZoneID(current.ID, current.StartByte, current.EndByte, current.Content)
			
		} else {
			merged = append(merged, current)
			current = next
		}
	}
	
	merged = append(merged, current)
	return merged
}

// SemanticMerger attempts to join lists, tables, and short paragraphs into unified semantic blocks.
type SemanticMerger struct {
	BaseMerger ZoneMerger
}

// Merge semantic applies heuristics to combine short zones that form logical lists.
func (sm SemanticMerger) Merge(zones []SignalZone) []SignalZone {
	if len(zones) == 0 {
		return nil
	}
	
	// First pass: spatial merge
	spatial := sm.BaseMerger.Merge(zones)
	if len(spatial) <= 1 {
		return spatial
	}
	
	var out []SignalZone
	current := spatial[0]
	
	for i := 1; i < len(spatial); i++ {
		next := spatial[i]
		
		// If current is short and next is short, they might be list items
		if current.TokenCount < 20 && next.TokenCount < 20 {
			current.Content = current.Content + "\n* " + next.Content
			current.EndByte = next.EndByte
			current.TokenCount += next.TokenCount
			current.ByteCount = len([]byte(current.Content))
			current.Density = (current.Density + next.Density) / 2.0 // Approx
			current.ID = stableZoneID("semantic", current.StartByte, current.EndByte, current.Content)
		} else {
			out = append(out, current)
			current = next
		}
	}
	out = append(out, current)
	return out
}

// ─── DOM Tree Heuristic Extraction ──────────────────────────────────────────

// DOMHeuristics defines weights for structural HTML signals.
type DOMHeuristics struct {
	TitleWeight   float64
	HeaderWeight  float64
	ArticleWeight float64
	FooterPenalty float64
	NavPenalty    float64
}

// ScoreDOMNode assigns a signal probability to an HTML block based on tag context.
func (h DOMHeuristics) ScoreDOMNode(tagName string, classNames string, textLength int) float64 {
	score := 1.0
	tag := strings.ToLower(tagName)
	classes := strings.ToLower(classNames)
	
	// Tag-based weights
	switch tag {
	case "h1", "title":
		score *= h.TitleWeight
	case "h2", "h3":
		score *= h.HeaderWeight
	case "article", "main":
		score *= h.ArticleWeight
	case "footer", "nav", "aside", "script", "style":
		score *= 0.1 // Heavy penalty
	}
	
	// Class-based heuristic adjustments
	if strings.Contains(classes, "comment") || strings.Contains(classes, "reply") {
		score *= 0.5
	}
	if strings.Contains(classes, "content") || strings.Contains(classes, "body") || strings.Contains(classes, "article") {
		score *= 1.5
	}
	if strings.Contains(classes, "sidebar") || strings.Contains(classes, "menu") {
		score *= h.NavPenalty
	}
	if strings.Contains(classes, "footer") || strings.Contains(classes, "bottom") {
		score *= h.FooterPenalty
	}
	
	// Length constraint: Tiny text in divs is usually UI, long text is signal
	if tag == "div" || tag == "span" {
		if textLength < 50 {
			score *= 0.4
		} else if textLength > 200 {
			score *= 1.2
		}
	}
	
	return clampFloat(score, 0.01, 2.0)
}

// ─── Regex Micro-Extractors ─────────────────────────────────────────────────

var (
	rxDate   = regexp.MustCompile(`(?i)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{4}\b|\b\d{4}-\d{2}-\d{2}\b`)
	rxEmail  = regexp.MustCompile(`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`)
	rxAuthor = regexp.MustCompile(`(?i)\b(?:By|Author)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b`)
)

// EntitySet holds structured facts extracted from a zone.
type EntitySet struct {
	Dates   []string `json:"dates,omitempty"`
	Emails  []string `json:"emails,omitempty"`
	Authors []string `json:"authors,omitempty"`
}

// ExtractEntities scans a zone for structured data using regex patterns.
func ExtractEntities(text string) EntitySet {
	if text == "" {
		return EntitySet{}
	}
	
	dates := rxDate.FindAllString(text, 5)
	for i, d := range dates {
		dates[i] = strings.TrimSpace(d)
	}
	
	emails := rxEmail.FindAllString(text, 5)
	
	var authors []string
	matches := rxAuthor.FindAllStringSubmatch(text, 3)
	for _, m := range matches {
		if len(m) > 1 {
			authors = append(authors, strings.TrimSpace(m[1]))
		}
	}
	
	return EntitySet{
		Dates:   dedupeStrings(dates),
		Emails:  dedupeStrings(emails),
		Authors: dedupeStrings(authors),
	}
}

func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]bool)
	out := make([]string, 0, len(in))
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// ─── Link Extraction & Categorization ───────────────────────────────────────

// ExtractedLink represents a hyperlink found within a signal zone.
type ExtractedLink struct {
	URL        string `json:"url"`
	AnchorText string `json:"anchor_text"`
	Category   string `json:"category"` // "internal", "external", "resource"
	NoFollow   bool   `json:"nofollow"`
}

// LinkExtractor parses hrefs and categorizes them relative to the base domain.
type LinkExtractor struct {
	BaseDomain string
}

// CategorizeLink determines the topological role of an extracted link.
func (le LinkExtractor) CategorizeLink(href string) string {
	href = strings.TrimSpace(href)
	if href == "" || strings.HasPrefix(href, "javascript:") || strings.HasPrefix(href, "mailto:") {
		return "ignored"
	}
	
	// Fast path for relative links
	if strings.HasPrefix(href, "/") && !strings.HasPrefix(href, "//") {
		if isResourceExtension(href) {
			return "resource"
		}
		return "internal"
	}
	
	parsed, err := url.Parse(href)
	if err != nil {
		return "unknown"
	}
	
	host := normalizeDomain(parsed.Host)
	if host == "" {
		return "internal" // Likely relative
	}
	
	if host == le.BaseDomain || strings.HasSuffix(host, "."+le.BaseDomain) {
		if isResourceExtension(parsed.Path) {
			return "resource"
		}
		return "internal"
	}
	
	return "external"
}

func isResourceExtension(path string) bool {
	lower := strings.ToLower(path)
	exts := []string{".jpg", ".png", ".gif", ".pdf", ".css", ".js", ".svg", ".zip", ".mp4"}
	for _, ext := range exts {
		if strings.HasSuffix(lower, ext) {
			return true
		}
	}
	return false
}

// ─── Language Tokenization Hooks ────────────────────────────────────────────

// Tokenizer defines a language-aware string splitter.
type Tokenizer interface {
	Tokenize(text string) []string
}

// EnglishTokenizer provides basic whitespace/punctuation tokenization.
type EnglishTokenizer struct{}

func (t EnglishTokenizer) Tokenize(text string) []string {
	if text == "" {
		return nil
	}
	f := func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}
	return strings.FieldsFunc(text, f)
}

// CJKTokenizer approximates token count for continuous scripts.
type CJKTokenizer struct{}

func (t CJKTokenizer) Tokenize(text string) []string {
	if text == "" {
		return nil
	}
	var tokens []string
	var current strings.Builder
	
	for _, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			continue
		}
		// In CJK, treat every ideograph as a separate token roughly
		if unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

// SelectTokenizer chooses the best tokenizer for a given language code.
func SelectTokenizer(langCode string) Tokenizer {
	langCode = strings.ToLower(strings.Split(langCode, "-")[0])
	switch langCode {
	case "zh", "ja", "ko":
		return CJKTokenizer{}
	default:
		return EnglishTokenizer{}
	}
}

// ─── Table Data Extraction ──────────────────────────────────────────────────

// TableExtraction represents structured grid data found in a zone.
type TableExtraction struct {
	Headers []string   `json:"headers"`
	Rows    [][]string `json:"rows"`
}

// TableExtractor attempts to convert text with grid-like formatting into structured rows.
type TableExtractor struct {
	MinRows int
	MinCols int
}

// Extract parses simple TSV, CSV, or markdown-style tables from raw text.
func (te TableExtractor) Extract(text string) *TableExtraction {
	if text == "" {
		return nil
	}
	
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) < te.MinRows {
		return nil
	}

	var rows [][]string
	delimiter := te.detectDelimiter(lines)
	if delimiter == "" {
		return nil // Not recognized as a table
	}

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || te.isSeparatorRow(line) {
			continue
		}
		
		// Remove leading/trailing markdown pipes if present
		if delimiter == "|" {
			line = strings.TrimPrefix(line, "|")
			line = strings.TrimSuffix(line, "|")
		}
		
		cells := strings.Split(line, delimiter)
		var cleaned []string
		for _, cell := range cells {
			cleaned = append(cleaned, strings.TrimSpace(cell))
		}
		
		if len(cleaned) >= te.MinCols {
			rows = append(rows, cleaned)
		}
	}

	if len(rows) < te.MinRows {
		return nil
	}

	// Assume first row is header
	return &TableExtraction{
		Headers: rows[0],
		Rows:    rows[1:],
	}
}

func (te TableExtractor) detectDelimiter(lines []string) string {
	if len(lines) < 2 {
		return ""
	}
	
	// Check first 3 data lines to find a consistent delimiter
	checkLines := lines
	if len(checkLines) > 3 {
		checkLines = checkLines[:3]
	}
	
	delims := []string{"|", "\t", ","}
	for _, d := range delims {
		consistent := true
		expected := -1
		
		for _, line := range checkLines {
			if te.isSeparatorRow(line) {
				continue
			}
			count := strings.Count(line, d)
			if count < te.MinCols-1 {
				consistent = false
				break
			}
			if expected == -1 {
				expected = count
			} else if count != expected {
				consistent = false
				break
			}
		}
		
		if consistent && expected > 0 {
			return d
		}
	}
	return ""
}

func (te TableExtractor) isSeparatorRow(line string) bool {
	// Matches markdown table separators like |---|---|
	clean := strings.ReplaceAll(line, " ", "")
	clean = strings.ReplaceAll(clean, "|", "")
	clean = strings.ReplaceAll(clean, "-", "")
	clean = strings.ReplaceAll(clean, ":", "")
	return len(clean) == 0 && strings.Contains(line, "-")
}

// ─── Image & Caption Pairing ────────────────────────────────────────────────

// ImageContext represents an image URL paired with surrounding descriptive text.
type ImageContext struct {
	ImageURL string `json:"image_url"`
	AltText  string `json:"alt_text"`
	Caption  string `json:"caption"`
	Score    float64 `json:"score"`
}

// PairImageCaptions looks for images in a zone and associates them with nearby text.
func PairImageCaptions(zone *SignalZone) []ImageContext {
	if zone == nil || zone.Content == "" {
		return nil
	}
	
	// Highly simplified mock of DOM parsing since we operate on text.
	// In reality, this relies on structural tags preserved as markdown or custom tokens.
	var results []ImageContext
	
	// Look for markdown image syntax: ![alt text](url)
	rxMarkdownImg := regexp.MustCompile(`!\[([^\]]*)\]\(([^)]+)\)`)
	matches := rxMarkdownImg.FindAllStringSubmatchIndex(zone.Content, -1)
	
	for _, m := range matches {
		if len(m) >= 6 {
			alt := zone.Content[m[2]:m[3]]
			url := zone.Content[m[4]:m[5]]
			
			// Look for caption text immediately following the image
			caption := ""
			afterIdx := m[1]
			if afterIdx < len(zone.Content) {
				// Get next line or up to 100 chars
				tail := zone.Content[afterIdx:]
				nl := strings.Index(tail, "\n")
				if nl > 0 && nl < 150 {
					caption = strings.TrimSpace(tail[:nl])
				}
			}
			
			score := 1.0
			if alt != "" {
				score += 0.5
			}
			if caption != "" {
				score += 1.0
			}
			
			results = append(results, ImageContext{
				ImageURL: url,
				AltText:  alt,
				Caption:  caption,
				Score:    score,
			})
		}
	}
	
	return results
}

// ─── Extraction Caching & Memoization ───────────────────────────────────────

// ExtractionCache stores processed zones to prevent redundant extraction.
type ExtractionCache struct {
	mu    sync.RWMutex
	store map[string]cachedExtraction
}

type cachedExtraction struct {
	Zones     []SignalZone
	Extracted int64
}

// NewExtractionCache creates an in-memory cache for extraction results.
func NewExtractionCache() *ExtractionCache {
	return &ExtractionCache{
		store: make(map[string]cachedExtraction),
	}
}

// Get retrieves cached zones if they are fresher than maxAge.
func (c *ExtractionCache) Get(url string, maxAge int64) ([]SignalZone, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	entry, ok := c.store[url]
	if !ok {
		return nil, false
	}
	
	if time.Now().Unix()-entry.Extracted > maxAge {
		return nil, false // Expired
	}
	
	return append([]SignalZone(nil), entry.Zones...), true
}

// Put stores extraction results in the cache.
func (c *ExtractionCache) Put(url string, zones []SignalZone) {
	if c == nil || url == "" || len(zones) == 0 {
		return
	}
	
	// Deep copy to prevent mutation
	cp := make([]SignalZone, len(zones))
	copy(cp, zones)
	
	c.mu.Lock()
	defer c.mu.Unlock()
	c.store[url] = cachedExtraction{
		Zones:     cp,
		Extracted: time.Now().Unix(),
	}
}

// Evict removes stale entries from the cache.
func (c *ExtractionCache) Evict(maxAge int64) int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	
	now := time.Now().Unix()
	count := 0
	for k, v := range c.store {
		if now-v.Extracted > maxAge {
			delete(c.store, k)
			count++
		}
	}
	return count
}

// ─── Signal Scoring Algorithms ──────────────────────────────────────────────

// SignalScorer evaluates the quality and density of a zone.
type SignalScorer struct {
	MinDensity       float64
	MinTokens        int
	OptimalTokenBand [2]int
}

// DefaultSignalScorer returns standard tuning for text density.
func DefaultSignalScorer() SignalScorer {
	return SignalScorer{
		MinDensity:       0.35,
		MinTokens:        15,
		OptimalTokenBand: [2]int{50, 500},
	}
}

// Score assigns a quality metric [0.0, 1.0] to a zone.
func (s SignalScorer) Score(zone *SignalZone) float64 {
	if zone == nil || zone.TokenCount < s.MinTokens || zone.Density < s.MinDensity {
		return 0.0
	}
	
	// Base score from density
	score := clampFloat((zone.Density-s.MinDensity)/(1.0-s.MinDensity), 0.1, 1.0)
	
	// Adjust for length: too short lacks context, too long might be a dump
	if zone.TokenCount < s.OptimalTokenBand[0] {
		// Penalty for being short
		ratio := float64(zone.TokenCount) / float64(s.OptimalTokenBand[0])
		score *= (0.5 + 0.5*ratio)
	} else if zone.TokenCount > s.OptimalTokenBand[1] {
		// Mild penalty for being overly long
		ratio := float64(s.OptimalTokenBand[1]) / float64(zone.TokenCount)
		score *= (0.8 + 0.2*ratio)
	}
	
	// Heuristic bonuses based on content features
	if strings.Contains(zone.Content, "?") && strings.Contains(zone.Content, ".") {
		score *= 1.1 // likely prose/Q&A
	}
	
	// Penalties for likely code/noise if not explicitly tagged
	if strings.Contains(zone.Content, "function(") || strings.Contains(zone.Content, "var ") {
		score *= 0.5
	}
	
	return clampFloat(score, 0.0, 1.0)
}

// TextToTagRatio approximates the amount of visible text vs HTML markup.
func TextToTagRatio(rawHTML string, extractedText string) float64 {
	rawBytes := len([]byte(rawHTML))
	textBytes := len([]byte(extractedText))
	
	if rawBytes == 0 {
		return 0.0
	}
	
	return clampFloat(float64(textBytes)/float64(rawBytes), 0.0, 1.0)
}

// ─── Batch Extraction Orchestration ─────────────────────────────────────────

// BatchExtractor orchestrates concurrent extraction across multiple payloads.
type BatchExtractor struct {
	Concurrency int
	Pipeline    *ExtractionPipeline
	Cache       *ExtractionCache
}

// ExtractionPayload represents input for the batch extractor.
type ExtractionPayload struct {
	URL           string
	RawContent    string
	TopologyClass string
}

// ExtractionResult represents the output for a single payload.
type ExtractionResult struct {
	URL   string
	Zones []SignalZone
	Error error
}

// NewBatchExtractor creates a new orchestrator.
func NewBatchExtractor(concurrency int, pipeline *ExtractionPipeline) *BatchExtractor {
	if concurrency <= 0 {
		concurrency = 4
	}
	return &BatchExtractor{
		Concurrency: concurrency,
		Pipeline:    pipeline,
		Cache:       NewExtractionCache(),
	}
}

// ProcessBatch runs extraction concurrently over a set of payloads.
func (be *BatchExtractor) ProcessBatch(payloads []ExtractionPayload) []ExtractionResult {
	if len(payloads) == 0 {
		return nil
	}

	results := make([]ExtractionResult, len(payloads))
	
	// Setup worker pool
	jobs := make(chan struct {
		Index   int
		Payload ExtractionPayload
	}, len(payloads))
	
	var wg sync.WaitGroup
	
	// Start workers
	for w := 0; w < be.Concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				// Check cache first
				if cached, ok := be.Cache.Get(job.Payload.URL, 3600); ok {
					results[job.Index] = ExtractionResult{
						URL:   job.Payload.URL,
						Zones: cached,
					}
					continue
				}
				
				// Extract
				zones, err := ExtractSignalZones(job.Payload.RawContent, job.Payload.URL)
				if err != nil {
					results[job.Index] = ExtractionResult{
						URL:   job.Payload.URL,
						Error: err,
					}
					continue
				}
				
				// Run pipeline on each zone
				var processed []SignalZone
				for _, z := range zones {
					// Copy zone to pass to pipeline
					zc := z
					if err := be.Pipeline.Run(&zc, job.Payload.TopologyClass); err == nil {
						processed = append(processed, zc)
					}
				}
				
				// Cache result
				be.Cache.Put(job.Payload.URL, processed)
				
				results[job.Index] = ExtractionResult{
					URL:   job.Payload.URL,
					Zones: processed,
				}
			}
		}()
	}

	// Dispatch jobs
	for i, p := range payloads {
		jobs <- struct {
			Index   int
			Payload ExtractionPayload
		}{i, p}
	}
	close(jobs)
	wg.Wait()
	
	return results
}

// ─── Extraction Statistics ──────────────────────────────────────────────────

// ExtractionStats provides metrics for a batch of extractions.
type ExtractionStats struct {
	TotalProcessed int     `json:"total_processed"`
	TotalErrors    int     `json:"total_errors"`
	TotalZones     int     `json:"total_zones"`
	MeanZones      float64 `json:"mean_zones"`
	CacheHitRate   float64 `json:"cache_hit_rate"`
}

// ComputeExtractionStats aggregates metrics from batch results.
func ComputeExtractionStats(results []ExtractionResult, cacheHits int) ExtractionStats {
	stats := ExtractionStats{
		TotalProcessed: len(results),
	}
	if stats.TotalProcessed == 0 {
		return stats
	}
	
	for _, r := range results {
		if r.Error != nil {
			stats.TotalErrors++
		} else {
			stats.TotalZones += len(r.Zones)
		}
	}
	
	successful := stats.TotalProcessed - stats.TotalErrors
	if successful > 0 {
		stats.MeanZones = float64(stats.TotalZones) / float64(successful)
	}
	
	stats.CacheHitRate = float64(cacheHits) / float64(stats.TotalProcessed)
	return stats
}

// ─── Simplified AST Representation ──────────────────────────────────────────

// DOMNode represents a lightweight AST node for heuristic extraction.
type DOMNode struct {
	Tag        string
	Classes    []string
	ID         string
	Attributes map[string]string
	Text       string
	Children   []*DOMNode
	Parent     *DOMNode
}

// NewDOMNode creates a fresh AST node.
func NewDOMNode(tag string) *DOMNode {
	return &DOMNode{
		Tag:        strings.ToLower(tag),
		Attributes: make(map[string]string),
	}
}

// AddChild appends a node to the children list.
func (n *DOMNode) AddChild(child *DOMNode) {
	if n == nil || child == nil {
		return
	}
	child.Parent = n
	n.Children = append(n.Children, child)
}

// HasClass checks if a node contains a specific CSS class.
func (n *DOMNode) HasClass(class string) bool {
	if n == nil {
		return false
	}
	class = strings.ToLower(class)
	for _, c := range n.Classes {
		if strings.ToLower(c) == class {
			return true
		}
	}
	return false
}

// TextContent recursively gathers all text from a node and its children.
func (n *DOMNode) TextContent() string {
	if n == nil {
		return ""
	}
	var sb strings.Builder
	if n.Text != "" {
		sb.WriteString(n.Text)
		sb.WriteString(" ")
	}
	for _, child := range n.Children {
		sb.WriteString(child.TextContent())
		sb.WriteString(" ")
	}
	return strings.TrimSpace(sb.String())
}

// FindNodesByTag searches the AST for nodes with the given tag.
func (n *DOMNode) FindNodesByTag(tag string) []*DOMNode {
	if n == nil {
		return nil
	}
	tag = strings.ToLower(tag)
	var found []*DOMNode
	if n.Tag == tag {
		found = append(found, n)
	}
	for _, child := range n.Children {
		found = append(found, child.FindNodesByTag(tag)...)
	}
	return found
}

// ─── HTML Entity & Unicode Normalization ────────────────────────────────────

var (
	htmlEntities = map[string]string{
		"&nbsp;": " ", "&lt;": "<", "&gt;": ">", "&amp;": "&",
		"&quot;": "\"", "&apos;": "'", "&cent;": "¢", "&pound;": "£",
		"&yen;": "¥", "&euro;": "€", "&copy;": "©", "&reg;": "®",
		"&#39;": "'", "&#34;": "\"", "&#38;": "&", "&#60;": "<", "&#62;": ">",
	}
	
	unicodeReplacements = map[rune]rune{
		0x2018: '\'', 0x2019: '\'', // Smart single quotes
		0x201C: '"',  0x201D: '"',  // Smart double quotes
		0x2013: '-',  0x2014: '-',  // En and Em dashes
		0x2026: '.',                // Ellipsis (replace with single dot or handle elsewhere)
		0x00A0: ' ',                // Non-breaking space
	}
)

// NormalizeExtractionText cleans up HTML entities and normalizes unicode punctuation.
func NormalizeExtractionText(text string) string {
	if text == "" {
		return ""
	}
	
	// Quick pass for entities
	for ent, val := range htmlEntities {
		if strings.Contains(text, ent) {
			text = strings.ReplaceAll(text, ent, val)
		}
	}
	
	// Handles numeric entities like &#123;
	text = decodeNumericEntities(text)
	
	// Unicode normalization pass
	var sb strings.Builder
	sb.Grow(len(text))
	for _, r := range text {
		if replacement, ok := unicodeReplacements[r]; ok {
			sb.WriteRune(replacement)
		} else if r == 0x2026 {
			sb.WriteString("...") // Ellipsis expansion
		} else {
			sb.WriteRune(r)
		}
	}
	
	return strings.TrimSpace(sb.String())
}

func decodeNumericEntities(text string) string {
	rx := regexp.MustCompile(`&#([0-9]+);`)
	return rx.ReplaceAllStringFunc(text, func(m string) string {
		numStr := m[2 : len(m)-1]
		if num, err := strconv.Atoi(numStr); err == nil && num > 0 && num < 0x10FFFF {
			return string(rune(num))
		}
		return m
	})
}

// ─── Zone Fingerprinting & Diffing ──────────────────────────────────────────

// ZoneFingerprint creates a fast structural hash of a zone's text.
func ZoneFingerprint(text string) uint32 {
	if text == "" {
		return 0
	}
	
	// Strip all whitespace and punctuation for a structural hash
	var sb strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			sb.WriteRune(unicode.ToLower(r))
		}
	}
	
	clean := sb.String()
	if clean == "" {
		return 0
	}
	
	// FNV-1a 32-bit
	var hash uint32 = 2166136261
	for i := 0; i < len(clean); i++ {
		hash ^= uint32(clean[i])
		hash *= 16777619
	}
	return hash
}

// ZoneDiff compares two arrays of zones and returns similarity metrics.
type ZoneDiff struct {
	SimilarityRatio float64 `json:"similarity_ratio"`
	AddedZones      int     `json:"added_zones"`
	RemovedZones    int     `json:"removed_zones"`
	StableZones     int     `json:"stable_zones"`
}

// CompareZones evaluates structural drift between two extractions.
func CompareZones(oldZones, newZones []SignalZone) ZoneDiff {
	if len(oldZones) == 0 && len(newZones) == 0 {
		return ZoneDiff{SimilarityRatio: 1.0}
	}
	if len(oldZones) == 0 || len(newZones) == 0 {
		return ZoneDiff{SimilarityRatio: 0.0, AddedZones: len(newZones), RemovedZones: len(oldZones)}
	}
	
	oldMap := make(map[uint32]bool)
	for _, z := range oldZones {
		oldMap[ZoneFingerprint(z.Content)] = true
	}
	
	newMap := make(map[uint32]bool)
	for _, z := range newZones {
		newMap[ZoneFingerprint(z.Content)] = true
	}
	
	stable := 0
	for fp := range newMap {
		if oldMap[fp] {
			stable++
		}
	}
	
	added := len(newMap) - stable
	removed := len(oldMap) - stable
	
	totalUnique := len(oldMap) + len(newMap) - stable
	similarity := float64(stable) / float64(totalUnique)
	
	return ZoneDiff{
		SimilarityRatio: clampFloat(similarity, 0, 1),
		AddedZones:      added,
		RemovedZones:    removed,
		StableZones:     stable,
	}
}

// ─── Heuristic Paragraph Segmentation ───────────────────────────────────────

// SegmentParagraphs breaks a large text block into logical paragraphs.
func SegmentParagraphs(text string) []string {
	if text == "" {
		return nil
	}
	
	// Standard double newline split
	blocks := strings.Split(text, "\n\n")
	var paras []string
	
	for _, b := range blocks {
		b = strings.TrimSpace(b)
		if b == "" {
			continue
		}
		
		// If a block is extremely long, attempt to break on sentence boundaries
		if len(b) > 2000 {
			subParas := breakLongParagraph(b)
			paras = append(paras, subParas...)
		} else {
			paras = append(paras, b)
		}
	}
	return paras
}

func breakLongParagraph(text string) []string {
	var paras []string
	var current strings.Builder
	
	sentences := splitSentences(text)
	for _, s := range sentences {
		current.WriteString(s)
		current.WriteString(" ")
		
		// Break arbitrarily around 1000 chars at a sentence boundary
		if current.Len() > 1000 {
			paras = append(paras, strings.TrimSpace(current.String()))
			current.Reset()
		}
	}
	
	if current.Len() > 0 {
		paras = append(paras, strings.TrimSpace(current.String()))
	}
	return paras
}

func splitSentences(text string) []string {
	// Crude sentence splitter: dot, question mark, or exclamation mark followed by space
	rx := regexp.MustCompile(`([.?!])\s+(?=[A-Z0-9])`)
	
	// The regex consumes the punctuation, so we need a different approach to keep it.
	// Alternative: find indices and substring.
	var sentences []string
	matches := rx.FindAllStringIndex(text, -1)
	
	start := 0
	for _, m := range matches {
		end := m[1] - 1 // keep the punctuation, drop the space
		sentences = append(sentences, strings.TrimSpace(text[start:end]))
		start = end
	}
	
	if start < len(text) {
		sentences = append(sentences, strings.TrimSpace(text[start:]))
	}
	
	return sentences
}

// ─── Extraction Confidence & Feature Vector Models ──────────────────────────

// FeatureVector represents mathematical attributes of a zone for ML confidence scoring.
type FeatureVector struct {
	TokenCount        int     `json:"f_token_count"`
	ByteCount         int     `json:"f_byte_count"`
	TextDensity       float64 `json:"f_text_density"`
	LinkDensity       float64 `json:"f_link_density"`
	ListDensity       float64 `json:"f_list_density"`
	StopwordRatio     float64 `json:"f_stopword_ratio"`
	PunctuationRatio  float64 `json:"f_punctuation_ratio"`
	AverageWordLength float64 `json:"f_avg_word_length"`
	Capitalization    float64 `json:"f_capitalization_ratio"`
}

// ExtractFeatureVector generates an ML-compatible feature vector from a zone.
func ExtractFeatureVector(text string, htmlTags string) FeatureVector {
	if text == "" {
		return FeatureVector{}
	}

	tokens := strings.FieldsFunc(text, func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	})
	
	tc := len(tokens)
	bc := len([]byte(text))
	
	// Basic counts
	stopwords := 0
	caps := 0
	wordLenSum := 0
	punctCount := 0
	
	// Stopwords list (minimal set for heuristic)
	swSet := map[string]bool{"the":true, "and":true, "of":true, "in":true, "to":true, "a":true, "is":true, "that":true, "it":true, "on":true}

	for _, t := range tokens {
		lower := strings.ToLower(t)
		if swSet[lower] {
			stopwords++
		}
		if len(t) > 0 && unicode.IsUpper(rune(t[0])) {
			caps++
		}
		wordLenSum += len([]byte(t))
	}
	
	for _, r := range text {
		if unicode.IsPunct(r) {
			punctCount++
		}
	}
	
	// Density metrics
	txtDen := density(text)
	linkDen := 0.0
	listDen := 0.0
	
	// Very crude heuristic for HTML tags if provided
	if htmlTags != "" {
		aTags := strings.Count(strings.ToLower(htmlTags), "<a ")
		liTags := strings.Count(strings.ToLower(htmlTags), "<li")
		
		if tc > 0 {
			linkDen = clampFloat(float64(aTags*3)/float64(tc), 0, 1)
			listDen = clampFloat(float64(liTags*5)/float64(tc), 0, 1)
		}
	}
	
	avgWordLen := 0.0
	if tc > 0 {
		avgWordLen = float64(wordLenSum) / float64(tc)
	}

	return FeatureVector{
		TokenCount:        tc,
		ByteCount:         bc,
		TextDensity:       txtDen,
		LinkDensity:       linkDen,
		ListDensity:       listDen,
		StopwordRatio:     clampFloat(float64(stopwords)/float64(tc+1), 0, 1),
		PunctuationRatio:  clampFloat(float64(punctCount)/float64(bc+1), 0, 1),
		AverageWordLength: avgWordLen,
		Capitalization:    clampFloat(float64(caps)/float64(tc+1), 0, 1),
	}
}

// ComputeConfidence uses a hardcoded logistic regression model over feature vectors.
func ComputeConfidence(features FeatureVector) float64 {
	// W: Arbitrary weights tuned for content extraction precision
	bias := -2.5
	wTokens := 0.001
	wDensity := 3.5
	wStopwords := 2.0
	wLinkPenalty := -4.0
	wListBonus := 0.5
	wPunct := -1.0
	
	z := bias +
		float64(features.TokenCount)*wTokens +
		features.TextDensity*wDensity +
		features.StopwordRatio*wStopwords +
		features.LinkDensity*wLinkPenalty +
		features.ListDensity*wListBonus +
		features.PunctuationRatio*wPunct

	// Sigmoid
	prob := 1.0 / (1.0 + math.Exp(-z))
	return clampFloat(prob, 0.01, 0.99)
}

// ─── Output Formatting Routines ─────────────────────────────────────────────

// FormatAsMarkdown converts a sequence of zones into a unified Markdown document.
func FormatAsMarkdown(zones []SignalZone, title string, url string) string {
	if len(zones) == 0 {
		return ""
	}
	
	var sb strings.Builder
	
	// Metadata header
	if title != "" {
		sb.WriteString("# ")
		sb.WriteString(title)
		sb.WriteString("\n\n")
	}
	if url != "" {
		sb.WriteString("> Source: ")
		sb.WriteString(url)
		sb.WriteString("\n\n")
	}
	
	// Zones
	for i, z := range zones {
		// Attempt to determine if this is a header vs paragraph
		// Crude heuristic: short line, title case, no end punctuation
		isHeader := false
		if z.TokenCount < 10 && len(z.Content) < 80 {
			if !strings.HasSuffix(z.Content, ".") && !strings.HasSuffix(z.Content, "?") && !strings.HasSuffix(z.Content, "!") {
				isHeader = true
			}
		}
		
		if isHeader {
			sb.WriteString("## ")
			sb.WriteString(z.Content)
			sb.WriteString("\n\n")
		} else {
			sb.WriteString(z.Content)
			sb.WriteString("\n\n")
		}
		
		// Optional: add a separator between disjoint zones
		if i < len(zones)-1 {
			gap := zones[i+1].StartByte - z.EndByte
			if gap > 1000 {
				sb.WriteString("---\n\n")
			}
		}
	}
	
	return strings.TrimSpace(sb.String())
}

// ZoneJSONLD represents a structured metadata wrapper for semantic extraction.
type ZoneJSONLD struct {
	Context     string `json:"@context"`
	Type        string `json:"@type"`
	URL         string `json:"url,omitempty"`
	Headline    string `json:"headline,omitempty"`
	ArticleBody string `json:"articleBody"`
	WordCount   int    `json:"wordCount"`
}

// FormatAsJSONLD wraps extracted zones in Schema.org Article format.
func FormatAsJSONLD(zones []SignalZone, url string, title string) ([]byte, error) {
	if len(zones) == 0 {
		return nil, errors.New("no zones to format")
	}
	
	var body strings.Builder
	wordCount := 0
	
	for _, z := range zones {
		body.WriteString(z.Content)
		body.WriteString("\n\n")
		wordCount += z.TokenCount
	}
	
	doc := ZoneJSONLD{
		Context:     "https://schema.org",
		Type:        "Article",
		URL:         url,
		Headline:    title,
		ArticleBody: strings.TrimSpace(body.String()),
		WordCount:   wordCount,
	}
	
	return json.MarshalIndent(doc, "", "  ")
}

// ConcatenateZones joins a slice of zones into a single string with double newlines.
func ConcatenateZones(zones []SignalZone) string {
	if len(zones) == 0 {
		return ""
	}
	var sb strings.Builder
	for i, z := range zones {
		sb.WriteString(z.Content)
		if i < len(zones)-1 {
			sb.WriteString("\n\n")
		}
	}
	return sb.String()
}

// ─── Document Structure Representation ──────────────────────────────────────

// HeadingNode represents a section heading in the document hierarchy.
type HeadingNode struct {
	Level    int            `json:"level"`
	Text     string         `json:"text"`
	Children []*HeadingNode `json:"children,omitempty"`
}

// DocumentOutline captures the hierarchical structure of headings.
type DocumentOutline struct {
	Title    string         `json:"title"`
	Headings []*HeadingNode `json:"headings"`
}

// BuildOutline infers a document hierarchy from an array of headings (H1-H6).
// headings input must be pairs of [level, text].
func BuildOutline(title string, headings []struct{ Level int; Text string }) DocumentOutline {
	outline := DocumentOutline{Title: title}
	if len(headings) == 0 {
		return outline
	}

	var root []*HeadingNode
	var stack []*HeadingNode

	for _, h := range headings {
		node := &HeadingNode{Level: h.Level, Text: h.Text}
		
		// Pop from stack until we find a parent with a strictly smaller level number (higher hierarchy)
		for len(stack) > 0 && stack[len(stack)-1].Level >= node.Level {
			stack = stack[:len(stack)-1]
		}
		
		if len(stack) == 0 {
			root = append(root, node)
		} else {
			parent := stack[len(stack)-1]
			parent.Children = append(parent.Children, node)
		}
		
		stack = append(stack, node)
	}
	
	outline.Headings = root
	return outline
}

// ─── Advanced Text Density Algorithms ───────────────────────────────────────

// DensityMetrics contains detailed statistical measures of text blocks.
type DensityMetrics struct {
	LineLengthVariance   float64
	AverageLineLength    float64
	PunctuationDensity   float64
	UppercaseRatio       float64
	SentenceLengthSpread float64
}

// ComputeDensityMetrics performs a deep statistical analysis of a text zone.
func ComputeDensityMetrics(text string) DensityMetrics {
	if text == "" {
		return DensityMetrics{}
	}
	
	lines := strings.Split(text, "\n")
	var lineLengths []float64
	sumLen := 0.0
	
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		ln := float64(len([]byte(l)))
		lineLengths = append(lineLengths, ln)
		sumLen += ln
	}
	
	if len(lineLengths) == 0 {
		return DensityMetrics{}
	}
	
	avgLen := sumLen / float64(len(lineLengths))
	
	var varianceSum float64
	for _, ln := range lineLengths {
		diff := ln - avgLen
		varianceSum += diff * diff
	}
	variance := varianceSum / float64(len(lineLengths))
	
	punctCount := 0
	upperCount := 0
	totalChars := 0
	
	for _, r := range text {
		if !unicode.IsSpace(r) {
			totalChars++
			if unicode.IsPunct(r) {
				punctCount++
			}
			if unicode.IsUpper(r) {
				upperCount++
			}
		}
	}
	
	punctDensity := 0.0
	upperRatio := 0.0
	if totalChars > 0 {
		punctDensity = float64(punctCount) / float64(totalChars)
		upperRatio = float64(upperCount) / float64(totalChars)
	}
	
	sentences := splitSentences(text)
	var slens []float64
	sSum := 0.0
	for _, s := range sentences {
		ln := float64(len(strings.Fields(s)))
		slens = append(slens, ln)
		sSum += ln
	}
	
	sVariance := 0.0
	if len(slens) > 0 {
		sAvg := sSum / float64(len(slens))
		for _, sl := range slens {
			d := sl - sAvg
			sVariance += d * d
		}
		sVariance /= float64(len(slens))
	}
	
	return DensityMetrics{
		LineLengthVariance:   variance,
		AverageLineLength:    avgLen,
		PunctuationDensity:   punctDensity,
		UppercaseRatio:       upperRatio,
		SentenceLengthSpread: sVariance,
	}
}

// ─── Content Sanitization Policies ──────────────────────────────────────────

// SanitizationPolicy defines rules for HTML stripping.
type SanitizationPolicy struct {
	AllowedTags       map[string]bool
	StripComments     bool
	CollapseWhitespace bool
	RemoveEmptyTags   bool
}

// StrictTextPolicy allows only fundamental formatting.
func StrictTextPolicy() SanitizationPolicy {
	return SanitizationPolicy{
		AllowedTags: map[string]bool{
			"p": true, "br": true, "strong": true, "em": true, "b": true, "i": true,
			"h1": true, "h2": true, "h3": true, "h4": true, "h5": true, "h6": true,
			"ul": true, "ol": true, "li": true, "blockquote": true, "a": true,
		},
		StripComments:      true,
		CollapseWhitespace: true,
		RemoveEmptyTags:    true,
	}
}

// MarkdownPolicy allows structure that maps well to Markdown.
func MarkdownPolicy() SanitizationPolicy {
	policy := StrictTextPolicy()
	policy.AllowedTags["img"] = true
	policy.AllowedTags["code"] = true
	policy.AllowedTags["pre"] = true
	policy.AllowedTags["hr"] = true
	policy.AllowedTags["table"] = true
	policy.AllowedTags["thead"] = true
	policy.AllowedTags["tbody"] = true
	policy.AllowedTags["tr"] = true
	policy.AllowedTags["th"] = true
	policy.AllowedTags["td"] = true
	return policy
}

// ─── Quality Filters & Gibberish Detection ──────────────────────────────────

// IsGibberish applies crude heuristics to detect base64 data, hex dumps, or line noise.
func IsGibberish(text string) bool {
	if len(text) < 50 {
		return false
	}
	
	// Fast check for long words (likely hashes or base64)
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return true
	}
	
	longWords := 0
	sumLen := 0
	for _, f := range fields {
		flen := len([]byte(f))
		sumLen += flen
		if flen > 30 && !strings.Contains(f, "http") {
			longWords++
		}
	}
	
	// If more than 10% of "words" are over 30 chars long, it's probably junk
	if float64(longWords)/float64(len(fields)) > 0.1 {
		return true
	}
	
	avgLen := float64(sumLen) / float64(len(fields))
	if avgLen > 25.0 {
		return true
	}
	
	// Check consonant clusters (English specific, but works okay as a general filter)
	consecutiveConsonants := 0
	maxConsonants := 0
	
	vowels := map[rune]bool{'a':true, 'e':true, 'i':true, 'o':true, 'u':true, 'y':true}
	
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) {
			if !vowels[r] {
				consecutiveConsonants++
				if consecutiveConsonants > maxConsonants {
					maxConsonants = consecutiveConsonants
				}
			} else {
				consecutiveConsonants = 0
			}
		} else {
			consecutiveConsonants = 0
		}
	}
	
	// More than 8 consecutive consonants is very rare in natural languages (even German/Polish)
	if maxConsonants > 8 {
		return true
	}
	
	return false
}

// FilterZones applies the gibberish detector to drop noise zones.
func FilterZones(zones []SignalZone) []SignalZone {
	var filtered []SignalZone
	for _, z := range zones {
		if !IsGibberish(z.Content) {
			filtered = append(filtered, z)
		}
	}
	return filtered
}

// ─── Extractor Logging & Diagnostics ────────────────────────────────────────

// ExtractorEvent records a diagnostic event during the extraction pipeline.
type ExtractorEvent struct {
	Timestamp int64  `json:"timestamp"`
	Plugin    string `json:"plugin"`
	Action    string `json:"action"`
	Message   string `json:"message"`
	Duration  int64  `json:"duration_ns"`
}

// ExtractionDiagnostics holds detailed tracing info for a single extraction run.
type ExtractionDiagnostics struct {
	URL          string           `json:"url"`
	InitialZones int              `json:"initial_zones"`
	FinalZones   int              `json:"final_zones"`
	Events       []ExtractorEvent `json:"events"`
	TotalTimeNS  int64            `json:"total_time_ns"`
}

// LogEvent appends a diagnostic event to the trace.
func (d *ExtractionDiagnostics) LogEvent(plugin, action, message string, duration int64) {
	if d == nil {
		return
	}
	d.Events = append(d.Events, ExtractorEvent{
		Timestamp: time.Now().UnixNano(),
		Plugin:    plugin,
		Action:    action,
		Message:   message,
		Duration:  duration,
	})
}

// ExtractSignalZones is a top-level function that breaks raw content into initial signal zones.
func ExtractSignalZones(content string, url string) ([]SignalZone, error) {
	if content == "" {
		return nil, errors.New("empty content")
	}
	// Basic heuristic extraction
	paragraphs := SegmentParagraphs(content)
	var zones []SignalZone
	offset := 0
	for _, p := range paragraphs {
		bc := len([]byte(p))
		tc := countTokens(p)
		z := SignalZone{
			ID:         stableZoneID(url, offset, offset+bc, p),
			Type:       ZoneProse,
			Content:    p,
			StartByte:  offset,
			EndByte:    offset + bc,
			TokenCount: tc,
			ByteCount:  bc,
			Density:    density(p),
		}
		zones = append(zones, z)
		offset += bc + 2 // +2 for double newline typical in SegmentParagraphs
	}
	return zones, nil
}
