package preparser

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
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
