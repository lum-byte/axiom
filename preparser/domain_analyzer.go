package preparser

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"hash/fnv"
	"io"
	"math"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const FingerprintSlotBytes = 256

const (
	TopologyGenericHTML       = "GENERIC_HTML"
	TopologyRESTAPIJSON       = "REST_API_JSON"
	TopologySaaSDocs          = "SAAS_DOCS"
	TopologyEcommerceProduct  = "ECOMMERCE_PRODUCT"
	TopologyNewsArticle       = "NEWS_ARTICLE"
	TopologyJSONLDStructured  = "JSON_LD_STRUCTURED"
	TopologySearchResultsPage = "SEARCH_RESULTS_PAGE"
	TopologyMediaGallery      = "MEDIA_GALLERY"
	TopologyAuthWall          = "AUTH_WALL"

	PhaseRecommendationCold     = "COLD"
	PhaseRecommendationLearning = "LEARNING"
	PhaseRecommendationKnown    = "KNOWN"
)

const (
	FrictionLevelCL1 = 1
	FrictionLevelCL2 = 2
	FrictionLevelCL3 = 3
	FrictionLevelCL4 = 4
)

type PathPattern struct {
	Pattern       string `json:"pattern"`
	TopologyClass string `json:"topology_class"`
}

type RateLimitProfile struct {
	Domain            string  `json:"domain"`
	RequestsPerSecond float64 `json:"requests_per_second"`
	CrawlDelaySeconds float64 `json:"crawl_delay_seconds"`
	BurstCapacity     int     `json:"burst_capacity"`
}

type FetchRecord struct {
	URL             string            `json:"url"`
	StatusCode      int               `json:"status_code"`
	ContentType     string            `json:"content_type"`
	ContentLanguage string            `json:"content_language"`
	ResponseBytes   int64             `json:"response_bytes"`
	LatencyMS       float64           `json:"latency_ms"`
	RedirectCount   int               `json:"redirect_count"`
	RobotsDenied    bool              `json:"robots_denied"`
	FetchedAtUnix    int64             `json:"fetched_at_unix"`
	TopologyHint    string            `json:"topology_hint"`
	Headers         map[string]string `json:"headers"`
}

type URLPattern struct {
	Pattern       string   `json:"pattern"`
	Count         int      `json:"count"`
	Depth         int      `json:"depth"`
	VariableSlots []int    `json:"variable_slots"`
	Examples      []string `json:"examples"`
	TopologyClass string   `json:"topology_class"`
	Confidence    float64  `json:"confidence"`
}

type RobotsRule struct {
	Directive string `json:"directive"`
	Path      string `json:"path"`
	Line      int    `json:"line"`
}

type RobotsAnalysis struct {
	HasRobots              bool         `json:"has_robots"`
	DisallowRules          []RobotsRule `json:"disallow_rules"`
	AllowRules             []RobotsRule `json:"allow_rules"`
	SitemapURLs            []string     `json:"sitemap_urls"`
	CrawlDelaySeconds      float64      `json:"crawl_delay_seconds"`
	DisallowDensity        float64      `json:"disallow_density"`
	FrictionLevel          int          `json:"friction_level"`
	BotMitigation          string       `json:"bot_mitigation"`
	RequiresClearance      bool         `json:"requires_clearance"`
	RobotsFingerprintHash  string       `json:"robots_fingerprint_hash"`
	PreferredFetchStrategy string       `json:"preferred_fetch_strategy"`
}

type OnlineStats struct {
	Count int64   `json:"count"`
	Mean  float64 `json:"mean"`
	M2    float64 `json:"m2"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
}

type FetchHealthSummary struct {
	Total               int               `json:"total"`
	Success             int               `json:"success"`
	ClientErrors         int               `json:"client_errors"`
	ServerErrors         int               `json:"server_errors"`
	Redirects           int               `json:"redirects"`
	RobotsDenied        int               `json:"robots_denied"`
	ByStatusClass       map[string]int    `json:"by_status_class"`
	ByContentTypeFamily map[string]int    `json:"by_content_type_family"`
	MedianLatencyMS     float64           `json:"median_latency_ms"`
	P95LatencyMS        float64           `json:"p95_latency_ms"`
	AverageBytes        float64           `json:"average_bytes"`
	LatestFetchUnix     int64             `json:"latest_fetch_unix"`
	ObservedLanguages   map[string]int    `json:"observed_languages"`
	RepresentativeURLs  map[string]string `json:"representative_urls"`
}

type DomainFingerprint struct {
	Domain                    string             `json:"domain"`
	TopologyClass             string             `json:"topology_class"`
	TopologyDistribution      map[string]float64 `json:"topology_distribution"`
	URLPatterns               []URLPattern       `json:"url_patterns"`
	RobotsSignals             RobotsAnalysis     `json:"robots_signals"`
	ContentLanguage           string             `json:"content_language"`
	AvgResponseSize           int64              `json:"avg_response_size"`
	AvgLatencyMS              float64            `json:"avg_latency_ms"`
	SizeStats                 OnlineStats        `json:"size_stats"`
	LatencyStats              OnlineStats        `json:"latency_stats"`
	Health                    FetchHealthSummary `json:"health"`
	PhaseRecommendation       string             `json:"phase_recommendation"`
	FrictionLevel             int                `json:"friction_level"`
	Confidence                float64            `json:"confidence"`
	ObservationCount          int                `json:"observation_count"`
	ObservedPathCount         int                `json:"observed_path_count"`
	SignalDensity             float64            `json:"signal_density"`
	FreshnessHalfLifeDays      float64            `json:"freshness_half_life_days"`
	FingerprintSHA256         string             `json:"fingerprint_sha256"`
	FingerprintSlot           []byte             `json:"fingerprint_slot"`
	AnalyzedAtUnix            int64              `json:"analyzed_at_unix"`
	RunID                     string             `json:"run_id"`
}

type RecordValidationIssue struct {
	URL      string `json:"url"`
	Field    string `json:"field"`
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
}

type FetchRecordValidationReport struct {
	Valid       bool                    `json:"valid"`
	IssueCount  int                     `json:"issue_count"`
	ErrorCount  int                     `json:"error_count"`
	WarnCount   int                     `json:"warn_count"`
	Issues      []RecordValidationIssue `json:"issues"`
	Total       int                     `json:"total"`
	Accepted    int                     `json:"accepted"`
	Rejected    int                     `json:"rejected"`
	DomainCount map[string]int          `json:"domain_count"`
}

type PatternDelta struct {
	Pattern       string  `json:"pattern"`
	PreviousCount int     `json:"previous_count"`
	CurrentCount  int     `json:"current_count"`
	ChangeRatio   float64 `json:"change_ratio"`
	Status        string  `json:"status"`
}

type TopologyDelta struct {
	TopologyClass string  `json:"topology_class"`
	PreviousShare float64 `json:"previous_share"`
	CurrentShare  float64 `json:"current_share"`
	Delta         float64 `json:"delta"`
}

type DomainDriftReport struct {
	Domain              string          `json:"domain"`
	PreviousHash        string          `json:"previous_hash"`
	CurrentHash         string          `json:"current_hash"`
	PatternDeltas       []PatternDelta  `json:"pattern_deltas"`
	TopologyDeltas      []TopologyDelta `json:"topology_deltas"`
	FrictionDelta       int             `json:"friction_delta"`
	ConfidenceDelta     float64         `json:"confidence_delta"`
	SignalDensityDelta  float64         `json:"signal_density_delta"`
	DriftScore          float64         `json:"drift_score"`
	DriftLevel          string          `json:"drift_level"`
	RecipeRefreshNeeded bool            `json:"recipe_refresh_needed"`
	PlanRefreshNeeded   bool            `json:"plan_refresh_needed"`
}

type LearningHint struct {
	Domain        string  `json:"domain"`
	Kind          string  `json:"kind"`
	Target        string  `json:"target"`
	Priority      float64 `json:"priority"`
	Reason        string  `json:"reason"`
	TopologyClass string  `json:"topology_class"`
}

type FingerprintValidationReport struct {
	Valid      bool     `json:"valid"`
	Errors     []string `json:"errors"`
	Warnings   []string `json:"warnings"`
	Domain     string   `json:"domain"`
	HashPresent bool     `json:"hash_present"`
}

type MemoryCursorStore struct {
	mu       sync.RWMutex
	history  map[string][]FetchRecord
	robots   map[string]string
	sitemaps map[string][]string
}

type CrawlURL struct {
	URL                 string           `json:"url"`
	TopologyHint        string           `json:"topology_hint"`
	FetchMode           string           `json:"fetch_mode"`
	RenderMode          string           `json:"render_mode"`
	Priority            int              `json:"priority"`
	RateLimitProfile    RateLimitProfile `json:"rate_limit_profile"`
	ExpectedContentType string           `json:"expected_content_type"`
	CrawlDelaySeconds   float64          `json:"crawl_delay_seconds"`
	MaxResponseBytes    int              `json:"max_response_bytes"`
	IsRobots            bool             `json:"is_robots"`
	IsSitemap           bool             `json:"is_sitemap"`
	RunID               string           `json:"run_id"`
}

type CrawlManifest struct {
	Domain                   string     `json:"domain"`
	URLs                     []CrawlURL `json:"urls"`
	TotalURLs                int        `json:"total_urls"`
	EstimatedDurationSeconds float64    `json:"estimated_duration_seconds"`
	ClearanceRequired        int        `json:"clearance_required"`
	ManifestID               string     `json:"manifest_id"`
}

type DomainMap struct {
	Domain                      string            `json:"domain"`
	DisallowedTopologyClasses   map[string]string `json:"disallowed_topology_classes"`
	AllowedSignalPaths          []PathPattern     `json:"allowed_signal_paths"`
	CrawlDelaySeconds           float64           `json:"crawl_delay_seconds"`
	SitemapURLs                 []string          `json:"sitemap_urls"`
	PathTopologyMap             map[string]string `json:"path_topology_map"`
	FrictionZones               []PathPattern     `json:"friction_zones"`
	SignalZones                 []PathPattern     `json:"signal_zones"`
	BotMitigation               string            `json:"bot_mitigation"`
	RenderRequirements          map[string]string `json:"render_requirements"`
	RateLimitProfile            RateLimitProfile  `json:"rate_limit_profile"`
	CrawlManifest               CrawlManifest     `json:"crawl_manifest"`
	ObservedPathCount           int               `json:"observed_path_count"`
	TopologyEntropyMilli        int               `json:"topology_entropy_milli"`
	FingerprintSHA256           string            `json:"fingerprint_sha256"`
	AnalyzedAtUnix              int64             `json:"analyzed_at_unix"`
}

type DomainTopologyEvent struct {
	Domain    string    `json:"domain"`
	DomainMap DomainMap `json:"domain_map"`
}

type BridgeRequest struct {
	Topic     string      `json:"topic"`
	Component string      `json:"component"`
	Payload   interface{} `json:"payload"`
}

type DomainAnalyzer struct {
	mu          sync.RWMutex
	fingerprints map[string][FingerprintSlotBytes]byte
}

func NewDomainAnalyzer() *DomainAnalyzer {
	return &DomainAnalyzer{fingerprints: make(map[string][FingerprintSlotBytes]byte)}
}

func (a *DomainAnalyzer) AnalyzeDomain(domain string, paths []string, robots string, sitemapURLs []string, runID string) (DomainMap, error) {
	domain = normalizeDomain(domain)
	if domain == "" {
		return DomainMap{}, errors.New("domain is empty")
	}
	if runID == "" {
		return DomainMap{}, errors.New("run_id is empty")
	}
	stats := analyzePaths(paths)
	delay := parseCrawlDelay(robots)
	rate := RateLimitProfile{
		Domain:            domain,
		RequestsPerSecond: requestsPerSecond(delay),
		CrawlDelaySeconds: delay,
		BurstCapacity:     burstForDelay(delay),
	}
	pathMap := map[string]string{}
	signals := make([]PathPattern, 0, len(stats.Patterns))
	for _, p := range stats.Patterns {
		tc := inferTopology(p)
		pathMap[p] = tc
		signals = append(signals, PathPattern{Pattern: p, TopologyClass: tc})
	}
	sort.Slice(signals, func(i, j int) bool { return signals[i].Pattern < signals[j].Pattern })
	manifest := buildSeedManifest(domain, sitemapURLs, rate, runID)
	fp := buildFingerprint(domain, paths, robots, sitemapURLs)
	m := DomainMap{
		Domain:                    domain,
		DisallowedTopologyClasses: disallowedFromRobots(robots),
		AllowedSignalPaths:        signals,
		CrawlDelaySeconds:         delay,
		SitemapURLs:               append([]string(nil), sitemapURLs...),
		PathTopologyMap:           pathMap,
		FrictionZones:             frictionFromRobots(robots),
		SignalZones:               signals,
		BotMitigation:             detectBotMitigation(robots),
		RenderRequirements:        renderRequirements(stats.Patterns),
		RateLimitProfile:          rate,
		CrawlManifest:             manifest,
		ObservedPathCount:         len(paths),
		TopologyEntropyMilli:      stats.EntropyMilli,
		FingerprintSHA256:         fingerprintHash(fp),
		AnalyzedAtUnix:            time.Now().Unix(),
	}
	a.mu.Lock()
	a.fingerprints[domain] = fp
	a.mu.Unlock()
	return m, nil
}

func (a *DomainAnalyzer) BatchAnalyze(items []DomainAnalysisInput) ([]DomainMap, error) {
	out := make([]DomainMap, 0, len(items))
	for _, item := range items {
		m, err := a.AnalyzeDomain(item.Domain, item.Paths, item.Robots, item.SitemapURLs, item.RunID)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, nil
}

type DomainAnalysisInput struct {
	Domain      string
	Paths       []string
	Robots      string
	SitemapURLs []string
	RunID       string
}

type CursorStore interface {
	ReadDomainHistory(domain string) ([]FetchRecord, error)
	ReadRobots(domain string) (string, error)
	ReadSitemaps(domain string) ([]string, error)
}

func (a *DomainAnalyzer) AnalyzeFetchRecords(domain string, history []FetchRecord, robots string, sitemapURLs []string, runID string) (*DomainFingerprint, error) {
	domain = normalizeDomain(domain)
	if domain == "" {
		return nil, errors.New("domain is empty")
	}
	if runID == "" {
		return nil, errors.New("run_id is empty")
	}
	normalizedHistory := filterDomainRecords(domain, history)
	if len(normalizedHistory) == 0 {
		return nil, errors.New("domain history is empty")
	}

	paths := make([]string, 0, len(normalizedHistory))
	topologyCounts := make(map[string]int)
	sizeStats := NewOnlineStats()
	latencyStats := NewOnlineStats()
	health := newFetchHealthSummary()
	languageCounts := make(map[string]int)
	latestFetch := int64(0)

	for _, record := range normalizedHistory {
		path := pathOf(record.URL)
		paths = append(paths, path)
		topology := strings.TrimSpace(record.TopologyHint)
		if topology == "" {
			topology = inferTopologyFromRecord(record)
		}
		topologyCounts[topology]++
		sizeStats.Add(float64(nonNegativeInt64(record.ResponseBytes)))
		latencyStats.Add(nonNegativeFloat(record.LatencyMS))
		health.Observe(record, topology)
		if record.FetchedAtUnix > latestFetch {
			latestFetch = record.FetchedAtUnix
		}
		language := normalizeLanguage(record.ContentLanguage, record.Headers)
		if language != "" {
			languageCounts[language]++
		}
	}

	patterns := BuildURLPatterns(paths, topologyCounts)
	robotsAnalysis := AnalyzeRobots(robots, sitemapURLs, len(paths))
	distribution := topologyDistribution(topologyCounts)
	dominantTopology := dominantDistributionKey(distribution, TopologyGenericHTML)
	confidence := fingerprintConfidence(len(normalizedHistory), distribution, robotsAnalysis, health)
	phase := phaseRecommendation(confidence, len(normalizedHistory), health)
	slot := BuildFingerprintSlot(FingerprintSlotInput{
		TopologyClass:       dominantTopology,
		FrictionLevel:       robotsAnalysis.FrictionLevel,
		PhaseRecommendation: phase,
		Confidence:          confidence,
		ObservationCount:    len(normalizedHistory),
	})
	fpHash := fingerprintHash(slot)
	contentLanguage := dominantLanguage(languageCounts)
	if contentLanguage == "" {
		contentLanguage = "unknown"
	}
	fingerprint := &DomainFingerprint{
		Domain:               domain,
		TopologyClass:        dominantTopology,
		TopologyDistribution: distribution,
		URLPatterns:          patterns,
		RobotsSignals:        robotsAnalysis,
		ContentLanguage:      contentLanguage,
		AvgResponseSize:      int64(math.Round(sizeStats.Mean)),
		AvgLatencyMS:         latencyStats.Mean,
		SizeStats:            sizeStats,
		LatencyStats:         latencyStats,
		Health:               health.Finalize(),
		PhaseRecommendation:  phase,
		FrictionLevel:        robotsAnalysis.FrictionLevel,
		Confidence:           confidence,
		ObservationCount:     len(normalizedHistory),
		ObservedPathCount:    len(paths),
		SignalDensity:        signalDensityFromPatterns(patterns, len(paths)),
		FreshnessHalfLifeDays: estimateFreshnessHalfLife(latestFetch, time.Now().Unix()),
		FingerprintSHA256:    fpHash,
		FingerprintSlot:      slot[:],
		AnalyzedAtUnix:       time.Now().Unix(),
		RunID:                runID,
	}
	a.mu.Lock()
	a.fingerprints[domain] = slot
	a.mu.Unlock()
	return fingerprint, nil
}

func (a *DomainAnalyzer) BatchAnalyzeStore(domains []string, store CursorStore, runID string) ([]*DomainFingerprint, error) {
	if store == nil {
		return nil, errors.New("cursor store is nil")
	}
	out := make([]*DomainFingerprint, 0, len(domains))
	for _, rawDomain := range domains {
		domain := normalizeDomain(rawDomain)
		if domain == "" {
			continue
		}
		history, err := store.ReadDomainHistory(domain)
		if err != nil {
			return nil, err
		}
		robots, err := store.ReadRobots(domain)
		if err != nil {
			return nil, err
		}
		sitemaps, err := store.ReadSitemaps(domain)
		if err != nil {
			return nil, err
		}
		fp, err := a.AnalyzeFetchRecords(domain, history, robots, sitemaps, runID)
		if err != nil {
			return nil, err
		}
		out = append(out, fp)
	}
	return out, nil
}

func ParseFetchRecordJSONL(r io.Reader) ([]FetchRecord, error) {
	if r == nil {
		return nil, errors.New("reader is nil")
	}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	records := make([]FetchRecord, 0, 128)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record FetchRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, errors.New("invalid fetch record jsonl at line " + strconv.Itoa(lineNumber) + ": " + err.Error())
		}
		records = append(records, record)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return records, nil
}

func SerializeDomainFingerprint(fp *DomainFingerprint) ([]byte, error) {
	if fp == nil {
		return nil, errors.New("fingerprint is nil")
	}
	return json.Marshal(fp)
}

func ValidateFetchRecord(record FetchRecord) []RecordValidationIssue {
	issues := make([]RecordValidationIssue, 0)
	addIssue := func(field, code, severity, message string) {
		issues = append(issues, RecordValidationIssue{URL: record.URL, Field: field, Code: code, Severity: severity, Message: message})
	}
	if strings.TrimSpace(record.URL) == "" {
		addIssue("url", "missing_url", "error", "fetch record URL is required")
	} else if parsed, err := url.Parse(record.URL); err != nil || parsed.Scheme == "" || parsed.Host == "" {
		addIssue("url", "invalid_url", "error", "fetch record URL must be absolute")
	} else if parsed.Scheme != "http" && parsed.Scheme != "https" {
		addIssue("url", "unsupported_scheme", "error", "fetch record URL must use http or https")
	}
	if record.StatusCode < 0 || record.StatusCode > 999 {
		addIssue("status_code", "invalid_status", "error", "status code is outside HTTP range")
	}
	if record.ResponseBytes < 0 {
		addIssue("response_bytes", "negative_size", "error", "response size cannot be negative")
	}
	if math.IsNaN(record.LatencyMS) || math.IsInf(record.LatencyMS, 0) || record.LatencyMS < 0 {
		addIssue("latency_ms", "invalid_latency", "error", "latency must be a non-negative finite number")
	}
	if record.StatusCode == 200 && record.ResponseBytes == 0 {
		addIssue("response_bytes", "empty_success", "warning", "successful response had zero bytes")
	}
	if record.ContentType == "" && record.StatusCode >= 200 && record.StatusCode < 300 {
		addIssue("content_type", "missing_content_type", "warning", "successful response has no content type")
	}
	if record.RedirectCount > 20 {
		addIssue("redirect_count", "redirect_loop_risk", "warning", "redirect count is unusually high")
	}
	return issues
}

func ValidateFetchRecords(records []FetchRecord) FetchRecordValidationReport {
	report := FetchRecordValidationReport{
		Valid:       true,
		DomainCount: make(map[string]int),
		Total:       len(records),
	}
	for _, record := range records {
		issues := ValidateFetchRecord(record)
		hasError := false
		for _, issue := range issues {
			report.Issues = append(report.Issues, issue)
			report.IssueCount++
			if issue.Severity == "error" {
				report.ErrorCount++
				hasError = true
			} else {
				report.WarnCount++
			}
		}
		if hasError {
			report.Rejected++
			report.Valid = false
			continue
		}
		report.Accepted++
		if parsed, err := url.Parse(record.URL); err == nil {
			domain := normalizeDomain(parsed.Host)
			if domain != "" {
				report.DomainCount[domain]++
			}
		}
	}
	return report
}

func CompareDomainFingerprints(previous *DomainFingerprint, current *DomainFingerprint) DomainDriftReport {
	if previous == nil && current == nil {
		return DomainDriftReport{DriftLevel: "unknown", DriftScore: 1}
	}
	if previous == nil {
		return DomainDriftReport{
			Domain:              current.Domain,
			CurrentHash:         current.FingerprintSHA256,
			DriftScore:          1,
			DriftLevel:          "new",
			RecipeRefreshNeeded: true,
			PlanRefreshNeeded:   true,
		}
	}
	if current == nil {
		return DomainDriftReport{
			Domain:              previous.Domain,
			PreviousHash:        previous.FingerprintSHA256,
			DriftScore:          1,
			DriftLevel:          "removed",
			RecipeRefreshNeeded: true,
			PlanRefreshNeeded:   true,
		}
	}
	patternDeltas := comparePatternSets(previous.URLPatterns, current.URLPatterns)
	topologyDeltas := compareTopologyDistribution(previous.TopologyDistribution, current.TopologyDistribution)
	frictionDelta := current.FrictionLevel - previous.FrictionLevel
	confidenceDelta := current.Confidence - previous.Confidence
	signalDelta := current.SignalDensity - previous.SignalDensity
	score := driftScore(patternDeltas, topologyDeltas, frictionDelta, confidenceDelta, signalDelta)
	level := driftLevel(score)
	return DomainDriftReport{
		Domain:              current.Domain,
		PreviousHash:        previous.FingerprintSHA256,
		CurrentHash:         current.FingerprintSHA256,
		PatternDeltas:       patternDeltas,
		TopologyDeltas:      topologyDeltas,
		FrictionDelta:       frictionDelta,
		ConfidenceDelta:     confidenceDelta,
		SignalDensityDelta:  signalDelta,
		DriftScore:          score,
		DriftLevel:          level,
		RecipeRefreshNeeded: level == "high" || level == "critical" || math.Abs(signalDelta) > 0.25,
		PlanRefreshNeeded:   level != "none" || frictionDelta != 0,
	}
}

func BuildLearningHints(fp *DomainFingerprint) []LearningHint {
	if fp == nil {
		return nil
	}
	hints := make([]LearningHint, 0)
	if fp.ObservationCount < 10 {
		hints = append(hints, LearningHint{
			Domain:        fp.Domain,
			Kind:          "expand_seed",
			Target:        fp.Domain,
			Priority:      0.85,
			Reason:        "insufficient observations for stable topology routing",
			TopologyClass: fp.TopologyClass,
		})
	}
	if fp.RobotsSignals.RequiresClearance {
		hints = append(hints, LearningHint{
			Domain:        fp.Domain,
			Kind:          "clearance_path",
			Target:        fp.Domain,
			Priority:      0.75 + float64(fp.RobotsSignals.FrictionLevel)/20.0,
			Reason:        "robots friction requires clearance-aware fetch scheduling",
			TopologyClass: fp.TopologyClass,
		})
	}
	for _, pattern := range fp.URLPatterns {
		if pattern.Confidence < 0.25 {
			continue
		}
		if pattern.TopologyClass == TopologyGenericHTML && pattern.Count >= 5 {
			hints = append(hints, LearningHint{
				Domain:        fp.Domain,
				Kind:          "classify_pattern",
				Target:        pattern.Pattern,
				Priority:      clampFloat(pattern.Confidence, 0.2, 1.0),
				Reason:        "high-volume pattern still classified as generic",
				TopologyClass: pattern.TopologyClass,
			})
		}
		if len(pattern.VariableSlots) > 0 && pattern.Count >= 3 {
			hints = append(hints, LearningHint{
				Domain:        fp.Domain,
				Kind:          "cursor_resume",
				Target:        pattern.Pattern,
				Priority:      clampFloat(0.4+pattern.Confidence/2, 0, 1),
				Reason:        "variable URL segment can become resumable cursor frontier",
				TopologyClass: pattern.TopologyClass,
			})
		}
	}
	sort.Slice(hints, func(i, j int) bool {
		if hints[i].Priority != hints[j].Priority {
			return hints[i].Priority > hints[j].Priority
		}
		return hints[i].Target < hints[j].Target
	})
	return hints
}

func ValidateDomainFingerprint(fp *DomainFingerprint) FingerprintValidationReport {
	report := FingerprintValidationReport{Valid: true}
	if fp == nil {
		return FingerprintValidationReport{Valid: false, Errors: []string{"fingerprint is nil"}}
	}
	report.Domain = fp.Domain
	report.HashPresent = fp.FingerprintSHA256 != ""
	if normalizeDomain(fp.Domain) == "" {
		report.Valid = false
		report.Errors = append(report.Errors, "domain is empty")
	}
	if fp.TopologyClass == "" {
		report.Valid = false
		report.Errors = append(report.Errors, "topology_class is empty")
	}
	if len(fp.TopologyDistribution) == 0 {
		report.Valid = false
		report.Errors = append(report.Errors, "topology_distribution is empty")
	}
	if fp.ObservationCount <= 0 {
		report.Valid = false
		report.Errors = append(report.Errors, "observation_count must be positive")
	}
	if fp.FrictionLevel < FrictionLevelCL1 || fp.FrictionLevel > FrictionLevelCL4 {
		report.Valid = false
		report.Errors = append(report.Errors, "friction_level outside CL1-CL4")
	}
	if fp.Confidence < 0 || fp.Confidence > 1 {
		report.Valid = false
		report.Errors = append(report.Errors, "confidence outside 0..1")
	}
	if fp.SignalDensity < 0 || fp.SignalDensity > 1 {
		report.Valid = false
		report.Errors = append(report.Errors, "signal_density outside 0..1")
	}
	if fp.FingerprintSHA256 == "" {
		report.Warnings = append(report.Warnings, "fingerprint hash missing")
	}
	if len(fp.URLPatterns) == 0 {
		report.Warnings = append(report.Warnings, "no URL patterns discovered")
	}
	if fp.RobotsSignals.FrictionLevel != 0 && fp.RobotsSignals.FrictionLevel != fp.FrictionLevel {
		report.Warnings = append(report.Warnings, "robots friction differs from fingerprint friction")
	}
	return report
}

func FingerprintSummaryLine(fp *DomainFingerprint) string {
	if fp == nil {
		return "fingerprint=nil"
	}
	parts := []string{
		"domain=" + fp.Domain,
		"topology=" + fp.TopologyClass,
		"phase=" + fp.PhaseRecommendation,
		"friction=CL" + strconv.Itoa(fp.FrictionLevel),
		"obs=" + strconv.Itoa(fp.ObservationCount),
		"confidence=" + strconv.FormatFloat(fp.Confidence, 'f', 3, 64),
		"signal_density=" + strconv.FormatFloat(fp.SignalDensity, 'f', 3, 64),
	}
	return strings.Join(parts, " ")
}

func NewMemoryCursorStore() *MemoryCursorStore {
	return &MemoryCursorStore{
		history:  make(map[string][]FetchRecord),
		robots:   make(map[string]string),
		sitemaps: make(map[string][]string),
	}
}

func (s *MemoryCursorStore) PutDomainHistory(domain string, records []FetchRecord) {
	if s == nil {
		return
	}
	domain = normalizeDomain(domain)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history[domain] = append([]FetchRecord(nil), records...)
}

func (s *MemoryCursorStore) PutRobots(domain string, robots string) {
	if s == nil {
		return
	}
	domain = normalizeDomain(domain)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.robots[domain] = robots
}

func (s *MemoryCursorStore) PutSitemaps(domain string, sitemaps []string) {
	if s == nil {
		return
	}
	domain = normalizeDomain(domain)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sitemaps[domain] = append([]string(nil), sitemaps...)
}

func (s *MemoryCursorStore) ReadDomainHistory(domain string) ([]FetchRecord, error) {
	if s == nil {
		return nil, errors.New("memory cursor store is nil")
	}
	domain = normalizeDomain(domain)
	s.mu.RLock()
	defer s.mu.RUnlock()
	records, ok := s.history[domain]
	if !ok {
		return nil, errors.New("domain history not found: " + domain)
	}
	return append([]FetchRecord(nil), records...), nil
}

func (s *MemoryCursorStore) ReadRobots(domain string) (string, error) {
	if s == nil {
		return "", errors.New("memory cursor store is nil")
	}
	domain = normalizeDomain(domain)
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.robots[domain], nil
}

func (s *MemoryCursorStore) ReadSitemaps(domain string) ([]string, error) {
	if s == nil {
		return nil, errors.New("memory cursor store is nil")
	}
	domain = normalizeDomain(domain)
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]string(nil), s.sitemaps[domain]...), nil
}

func (fp DomainFingerprint) ToDomainMap() DomainMap {
	pathMap := make(map[string]string, len(fp.URLPatterns))
	signalZones := make([]PathPattern, 0, len(fp.URLPatterns))
	for _, pattern := range fp.URLPatterns {
		pathMap[pattern.Pattern] = pattern.TopologyClass
		if pattern.Confidence >= 0.35 {
			signalZones = append(signalZones, PathPattern{Pattern: pattern.Pattern, TopologyClass: pattern.TopologyClass})
		}
	}
	sort.Slice(signalZones, func(i, j int) bool { return signalZones[i].Pattern < signalZones[j].Pattern })
	disallowed := make(map[string]string, len(fp.RobotsSignals.DisallowRules))
	for _, rule := range fp.RobotsSignals.DisallowRules {
		disallowed[rule.Path] = inferTopology(rule.Path)
	}
	return DomainMap{
		Domain:                    fp.Domain,
		DisallowedTopologyClasses: disallowed,
		AllowedSignalPaths:        signalZones,
		CrawlDelaySeconds:         fp.RobotsSignals.CrawlDelaySeconds,
		SitemapURLs:               append([]string(nil), fp.RobotsSignals.SitemapURLs...),
		PathTopologyMap:           pathMap,
		FrictionZones:             frictionRulesToPatterns(fp.RobotsSignals.DisallowRules),
		SignalZones:               signalZones,
		BotMitigation:             fp.RobotsSignals.BotMitigation,
		RenderRequirements:        renderRequirementsFromPatterns(fp.URLPatterns, fp.RobotsSignals),
		RateLimitProfile: RateLimitProfile{
			Domain:            fp.Domain,
			RequestsPerSecond: requestsPerSecond(fp.RobotsSignals.CrawlDelaySeconds),
			CrawlDelaySeconds: fp.RobotsSignals.CrawlDelaySeconds,
			BurstCapacity:     burstForDelay(fp.RobotsSignals.CrawlDelaySeconds),
		},
		ObservedPathCount:      fp.ObservedPathCount,
		TopologyEntropyMilli:   distributionEntropyMilli(fp.TopologyDistribution),
		FingerprintSHA256:      fp.FingerprintSHA256,
		AnalyzedAtUnix:         fp.AnalyzedAtUnix,
	}
}

func (fp DomainFingerprint) BridgeEvent() BridgeRequest {
	domainMap := fp.ToDomainMap()
	return BridgeRequest{
		Topic:     "domain_topology",
		Component: "preparser.domain_analyzer",
		Payload:   DomainTopologyEvent{Domain: fp.Domain, DomainMap: domainMap},
	}
}

func (a *DomainAnalyzer) ReadFingerprint(domain string) ([FingerprintSlotBytes]byte, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	fp, ok := a.fingerprints[normalizeDomain(domain)]
	return fp, ok
}

func (m DomainMap) BridgeEvent() BridgeRequest {
	return BridgeRequest{
		Topic:     "domain_topology",
		Component: "preparser.domain_analyzer",
		Payload:   DomainTopologyEvent{Domain: m.Domain, DomainMap: m},
	}
}

func EncodeBridgeRequest(req BridgeRequest) ([]byte, error) {
	return json.Marshal(req)
}

type FingerprintSlotInput struct {
	TopologyClass       string
	FrictionLevel       int
	PhaseRecommendation string
	Confidence          float64
	ObservationCount    int
}

func BuildFingerprintSlot(input FingerprintSlotInput) [FingerprintSlotBytes]byte {
	var fp [FingerprintSlotBytes]byte
	fp[0] = topologyClassIndex(input.TopologyClass)
	fp[1] = byte(clampInt(input.FrictionLevel, FrictionLevelCL1, FrictionLevelCL4))
	fp[2] = phaseRecommendationIndex(input.PhaseRecommendation)
	conf := uint32(clampFloat(input.Confidence, 0, 1) * 1_000_000)
	binary.LittleEndian.PutUint32(fp[3:7], conf)
	binary.LittleEndian.PutUint32(fp[7:11], uint32(clampInt(input.ObservationCount, 0, 2147483647)))
	copy(fp[16:48], []byte(input.TopologyClass))
	copy(fp[48:80], []byte(input.PhaseRecommendation))
	return fp
}

func BuildURLPatterns(paths []string, topologyCounts map[string]int) []URLPattern {
	trie := NewPatternTrie()
	for _, raw := range paths {
		trie.Insert(normalizePath(raw))
	}
	patterns := trie.Compress()
	dominant := dominantCountKey(topologyCounts, TopologyGenericHTML)
	for i := range patterns {
		if patterns[i].TopologyClass == "" {
			patterns[i].TopologyClass = inferTopology(patterns[i].Pattern)
			if patterns[i].TopologyClass == TopologyGenericHTML && dominant != "" {
				patterns[i].TopologyClass = dominant
			}
		}
		if patterns[i].Count > 0 && len(paths) > 0 {
			patterns[i].Confidence = math.Sqrt(float64(patterns[i].Count) / float64(len(paths)))
		}
		if patterns[i].Confidence > 1 {
			patterns[i].Confidence = 1
		}
	}
	sort.Slice(patterns, func(i, j int) bool {
		if patterns[i].Count != patterns[j].Count {
			return patterns[i].Count > patterns[j].Count
		}
		return patterns[i].Pattern < patterns[j].Pattern
	})
	return patterns
}

type PatternTrie struct {
	root *patternTrieNode
}

type patternTrieNode struct {
	Segment  string
	Count    int
	Children map[string]*patternTrieNode
	Examples []string
}

func NewPatternTrie() *PatternTrie {
	return &PatternTrie{root: &patternTrieNode{Children: make(map[string]*patternTrieNode)}}
}

func (t *PatternTrie) Insert(path string) {
	if t == nil {
		return
	}
	path = normalizePath(path)
	if path == "" {
		return
	}
	segments := splitPathSegments(path)
	node := t.root
	node.Count++
	node.addExample(path)
	for _, segment := range segments {
		key := normalizeTrieSegment(segment)
		child := node.Children[key]
		if child == nil {
			child = &patternTrieNode{Segment: key, Children: make(map[string]*patternTrieNode)}
			node.Children[key] = child
		}
		child.Count++
		child.addExample(path)
		node = child
	}
}

func (t *PatternTrie) Compress() []URLPattern {
	if t == nil || t.root == nil {
		return nil
	}
	out := make([]URLPattern, 0)
	children := sortedTrieChildren(t.root)
	for _, child := range children {
		compressTrieNode(child, []string{}, &out)
	}
	if len(out) == 0 && t.root.Count > 0 {
		out = append(out, URLPattern{Pattern: "/", Count: t.root.Count, Depth: 0, Examples: append([]string(nil), t.root.Examples...), TopologyClass: TopologyGenericHTML, Confidence: 1})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Pattern == out[j].Pattern {
			return out[i].Count > out[j].Count
		}
		return out[i].Pattern < out[j].Pattern
	})
	return mergeURLPatterns(out)
}

func (n *patternTrieNode) addExample(path string) {
	if n == nil || path == "" {
		return
	}
	if len(n.Examples) >= 3 {
		return
	}
	for _, existing := range n.Examples {
		if existing == path {
			return
		}
	}
	n.Examples = append(n.Examples, path)
}

func compressTrieNode(node *patternTrieNode, prefix []string, out *[]URLPattern) {
	if node == nil {
		return
	}
	current := append(append([]string(nil), prefix...), node.Segment)
	children := sortedTrieChildren(node)
	if len(children) == 0 || shouldEmitTrieNode(node, children) {
		pattern, variableSlots := patternFromSegments(current)
		*out = append(*out, URLPattern{
			Pattern:       pattern,
			Count:         node.Count,
			Depth:         len(current),
			VariableSlots: variableSlots,
			Examples:      append([]string(nil), node.Examples...),
			TopologyClass: inferTopology(pattern),
			Confidence:    0,
		})
	}
	for _, child := range children {
		compressTrieNode(child, current, out)
	}
}

func shouldEmitTrieNode(node *patternTrieNode, children []*patternTrieNode) bool {
	if node == nil {
		return false
	}
	if len(children) == 0 {
		return true
	}
	if node.Count >= 3 && len(children) >= 3 {
		return true
	}
	for _, child := range children {
		if child.Count*2 >= node.Count {
			return false
		}
	}
	return node.Count >= 2
}

func sortedTrieChildren(node *patternTrieNode) []*patternTrieNode {
	if node == nil || len(node.Children) == 0 {
		return nil
	}
	keys := make([]string, 0, len(node.Children))
	for key := range node.Children {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]*patternTrieNode, 0, len(keys))
	for _, key := range keys {
		out = append(out, node.Children[key])
	}
	return out
}

func patternFromSegments(segments []string) (string, []int) {
	if len(segments) == 0 {
		return "/", nil
	}
	varSlots := make([]int, 0)
	parts := make([]string, 0, len(segments))
	for i, segment := range segments {
		if segment == "" {
			continue
		}
		if isVariableSegment(segment) {
			parts = append(parts, "*")
			varSlots = append(varSlots, i)
		} else {
			parts = append(parts, segment)
		}
	}
	if len(parts) == 0 {
		return "/", varSlots
	}
	return "/" + strings.Join(parts, "/") + "/*", varSlots
}

func mergeURLPatterns(patterns []URLPattern) []URLPattern {
	byPattern := make(map[string]URLPattern, len(patterns))
	for _, pattern := range patterns {
		existing, ok := byPattern[pattern.Pattern]
		if !ok {
			byPattern[pattern.Pattern] = pattern
			continue
		}
		existing.Count += pattern.Count
		existing.Examples = mergeExamples(existing.Examples, pattern.Examples, 5)
		if pattern.Depth < existing.Depth {
			existing.Depth = pattern.Depth
		}
		if pattern.Confidence > existing.Confidence {
			existing.Confidence = pattern.Confidence
		}
		byPattern[pattern.Pattern] = existing
	}
	out := make([]URLPattern, 0, len(byPattern))
	for _, pattern := range byPattern {
		out = append(out, pattern)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Pattern < out[j].Pattern
	})
	return out
}

func mergeExamples(a, b []string, limit int) []string {
	if limit <= 0 {
		return nil
	}
	out := make([]string, 0, limit)
	seen := make(map[string]bool)
	for _, set := range [][]string{a, b} {
		for _, item := range set {
			if item == "" || seen[item] {
				continue
			}
			out = append(out, item)
			seen[item] = true
			if len(out) >= limit {
				return out
			}
		}
	}
	return out
}

func splitPathSegments(path string) []string {
	path = strings.Trim(normalizePath(path), "/")
	if path == "" {
		return nil
	}
	raw := strings.Split(path, "/")
	out := make([]string, 0, len(raw))
	for _, segment := range raw {
		if segment == "" {
			continue
		}
		out = append(out, segment)
	}
	return out
}

func normalizeTrieSegment(segment string) string {
	segment = strings.TrimSpace(strings.ToLower(segment))
	if segment == "" {
		return ""
	}
	if isUUIDLike(segment) || isNumericLike(segment) || isHashLike(segment) {
		return "*"
	}
	if strings.Contains(segment, "?") {
		segment = strings.Split(segment, "?")[0]
	}
	if len(segment) > 48 {
		return "*"
	}
	return segment
}

func isVariableSegment(segment string) bool {
	return segment == "*" || isUUIDLike(segment) || isNumericLike(segment) || isHashLike(segment)
}

func isNumericLike(segment string) bool {
	if segment == "" {
		return false
	}
	digits := 0
	for _, r := range segment {
		if r >= '0' && r <= '9' {
			digits++
			continue
		}
		if r == '-' || r == '_' {
			continue
		}
		return false
	}
	return digits >= 2
}

func isHashLike(segment string) bool {
	if len(segment) < 12 {
		return false
	}
	hexCount := 0
	for _, r := range segment {
		switch {
		case r >= '0' && r <= '9':
			hexCount++
		case r >= 'a' && r <= 'f':
			hexCount++
		case r >= 'A' && r <= 'F':
			hexCount++
		default:
			return false
		}
	}
	return hexCount == len(segment)
}

func isUUIDLike(segment string) bool {
	if len(segment) != 36 {
		return false
	}
	for i, r := range segment {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if r != '-' {
				return false
			}
			continue
		}
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

func AnalyzeRobots(robots string, sitemapURLs []string, observedPathCount int) RobotsAnalysis {
	lines := strings.Split(robots, "\n")
	disallow := make([]RobotsRule, 0)
	allow := make([]RobotsRule, 0)
	sitemaps := canonicalSitemaps(sitemapURLs)
	for i, rawLine := range lines {
		line := stripRobotsComment(strings.TrimSpace(rawLine))
		if line == "" {
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)
		switch key {
		case "disallow":
			path := normalizePath(value)
			if path != "" {
				disallow = append(disallow, RobotsRule{Directive: "disallow", Path: path, Line: i + 1})
			}
		case "allow":
			path := normalizePath(value)
			if path != "" {
				allow = append(allow, RobotsRule{Directive: "allow", Path: path, Line: i + 1})
			}
		case "sitemap":
			if value != "" {
				sitemaps = appendUniqueString(sitemaps, value)
			}
		}
	}
	sortRobotsRules(disallow)
	sortRobotsRules(allow)
	delay := parseCrawlDelay(robots)
	density := 0.0
	if observedPathCount > 0 {
		density = float64(len(disallow)) / float64(observedPathCount)
	}
	mitigation := detectBotMitigation(robots)
	friction := frictionLevelFromRobots(delay, density, mitigation, len(sitemaps))
	strategy := "static"
	if friction >= FrictionLevelCL3 || mitigation != "none" {
		strategy = "clearance"
	}
	return RobotsAnalysis{
		HasRobots:              strings.TrimSpace(robots) != "",
		DisallowRules:          disallow,
		AllowRules:             allow,
		SitemapURLs:            sitemaps,
		CrawlDelaySeconds:      delay,
		DisallowDensity:        density,
		FrictionLevel:          friction,
		BotMitigation:          mitigation,
		RequiresClearance:      friction >= FrictionLevelCL3,
		RobotsFingerprintHash:  hashString(robots + strings.Join(sitemaps, "\x00")),
		PreferredFetchStrategy: strategy,
	}
}

func stripRobotsComment(line string) string {
	if idx := strings.Index(line, "#"); idx >= 0 {
		line = line[:idx]
	}
	return strings.TrimSpace(line)
}

func canonicalSitemaps(in []string) []string {
	out := make([]string, 0, len(in))
	for _, sm := range in {
		sm = strings.TrimSpace(sm)
		if sm == "" {
			continue
		}
		out = appendUniqueString(out, sm)
	}
	sort.Strings(out)
	return out
}

func appendUniqueString(in []string, item string) []string {
	for _, existing := range in {
		if existing == item {
			return in
		}
	}
	return append(in, item)
}

func sortRobotsRules(rules []RobotsRule) {
	sort.Slice(rules, func(i, j int) bool {
		if rules[i].Path == rules[j].Path {
			return rules[i].Line < rules[j].Line
		}
		return rules[i].Path < rules[j].Path
	})
}

func frictionLevelFromRobots(delay float64, density float64, mitigation string, sitemapCount int) int {
	level := FrictionLevelCL1
	if delay >= 0.5 || density >= 0.10 {
		level = FrictionLevelCL2
	}
	if delay >= 2.0 || density >= 0.30 || mitigation != "none" {
		level = FrictionLevelCL3
	}
	if delay >= 10.0 || density >= 0.60 || mitigation == "cloudflare" {
		level = FrictionLevelCL4
	}
	if sitemapCount > 0 && level > FrictionLevelCL1 {
		level--
	}
	return clampInt(level, FrictionLevelCL1, FrictionLevelCL4)
}

func NewOnlineStats() OnlineStats {
	return OnlineStats{Min: math.Inf(1), Max: math.Inf(-1)}
}

func (s *OnlineStats) Add(v float64) {
	if s == nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return
	}
	s.Count++
	if v < s.Min {
		s.Min = v
	}
	if v > s.Max {
		s.Max = v
	}
	delta := v - s.Mean
	s.Mean += delta / float64(s.Count)
	delta2 := v - s.Mean
	s.M2 += delta * delta2
}

func (s OnlineStats) Variance() float64 {
	if s.Count < 2 {
		return 0
	}
	return s.M2 / float64(s.Count-1)
}

func (s OnlineStats) StdDev() float64 {
	return math.Sqrt(s.Variance())
}

func (s OnlineStats) MarshalJSON() ([]byte, error) {
	type wire struct {
		Count    int64   `json:"count"`
		Mean     float64 `json:"mean"`
		Variance float64 `json:"variance"`
		StdDev   float64 `json:"std_dev"`
		Min      float64 `json:"min"`
		Max      float64 `json:"max"`
	}
	min := s.Min
	max := s.Max
	if s.Count == 0 {
		min = 0
		max = 0
	}
	return json.Marshal(wire{Count: s.Count, Mean: s.Mean, Variance: s.Variance(), StdDev: s.StdDev(), Min: min, Max: max})
}

type fetchHealthAccumulator struct {
	health    FetchHealthSummary
	latencies []float64
	bytes     OnlineStats
}

func newFetchHealthSummary() *fetchHealthAccumulator {
	return &fetchHealthAccumulator{
		health: FetchHealthSummary{
			ByStatusClass:       make(map[string]int),
			ByContentTypeFamily: make(map[string]int),
			ObservedLanguages:   make(map[string]int),
			RepresentativeURLs:  make(map[string]string),
		},
		bytes: NewOnlineStats(),
	}
}

func (a *fetchHealthAccumulator) Observe(record FetchRecord, topology string) {
	if a == nil {
		return
	}
	a.health.Total++
	statusClass := statusClass(record.StatusCode)
	a.health.ByStatusClass[statusClass]++
	switch statusClass {
	case "2xx":
		a.health.Success++
	case "3xx":
		a.health.Redirects++
	case "4xx":
		a.health.ClientErrors++
	case "5xx":
		a.health.ServerErrors++
	}
	if record.RobotsDenied {
		a.health.RobotsDenied++
	}
	family := contentTypeFamily(record.ContentType)
	a.health.ByContentTypeFamily[family]++
	if _, ok := a.health.RepresentativeURLs[topology]; !ok && record.URL != "" {
		a.health.RepresentativeURLs[topology] = record.URL
	}
	if record.LatencyMS >= 0 {
		a.latencies = append(a.latencies, record.LatencyMS)
	}
	a.bytes.Add(float64(nonNegativeInt64(record.ResponseBytes)))
	if record.FetchedAtUnix > a.health.LatestFetchUnix {
		a.health.LatestFetchUnix = record.FetchedAtUnix
	}
	language := normalizeLanguage(record.ContentLanguage, record.Headers)
	if language != "" {
		a.health.ObservedLanguages[language]++
	}
}

func (a *fetchHealthAccumulator) Finalize() FetchHealthSummary {
	if a == nil {
		return FetchHealthSummary{}
	}
	cp := a.health
	sort.Float64s(a.latencies)
	cp.MedianLatencyMS = percentileSorted(a.latencies, 0.50)
	cp.P95LatencyMS = percentileSorted(a.latencies, 0.95)
	cp.AverageBytes = a.bytes.Mean
	return cp
}

func percentileSorted(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	if p <= 0 {
		return vals[0]
	}
	if p >= 1 {
		return vals[len(vals)-1]
	}
	pos := p * float64(len(vals)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return vals[lo]
	}
	weight := pos - float64(lo)
	return vals[lo]*(1-weight) + vals[hi]*weight
}

func statusClass(status int) string {
	switch {
	case status >= 200 && status < 300:
		return "2xx"
	case status >= 300 && status < 400:
		return "3xx"
	case status >= 400 && status < 500:
		return "4xx"
	case status >= 500 && status < 600:
		return "5xx"
	default:
		return "unknown"
	}
}

func contentTypeFamily(contentType string) string {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	if contentType == "" {
		return "unknown"
	}
	if strings.Contains(contentType, "json") {
		return "json"
	}
	if strings.Contains(contentType, "html") {
		return "html"
	}
	if strings.Contains(contentType, "xml") {
		return "xml"
	}
	if strings.HasPrefix(contentType, "image/") {
		return "image"
	}
	if strings.HasPrefix(contentType, "text/") {
		return "text"
	}
	return strings.Split(contentType, ";")[0]
}

func filterDomainRecords(domain string, history []FetchRecord) []FetchRecord {
	out := make([]FetchRecord, 0, len(history))
	for _, record := range history {
		if record.URL == "" {
			continue
		}
		parsed, err := url.Parse(record.URL)
		if err != nil {
			continue
		}
		host := normalizeDomain(parsed.Host)
		if host == domain || strings.HasSuffix(host, "."+domain) {
			out = append(out, record)
		}
	}
	return out
}

func inferTopologyFromRecord(record FetchRecord) string {
	contentType := strings.ToLower(record.ContentType)
	path := pathOf(record.URL)
	switch {
	case strings.Contains(contentType, "json"):
		if strings.Contains(path, "schema") || strings.Contains(path, "ld") {
			return TopologyJSONLDStructured
		}
		return TopologyRESTAPIJSON
	case strings.Contains(path, "login") || strings.Contains(path, "signin") || record.StatusCode == 401 || record.StatusCode == 403:
		return TopologyAuthWall
	case strings.Contains(path, "search") || strings.Contains(path, "query"):
		return TopologySearchResultsPage
	case strings.Contains(contentType, "image") || strings.Contains(path, "gallery"):
		return TopologyMediaGallery
	default:
		return inferTopology(path)
	}
}

func topologyDistribution(counts map[string]int) map[string]float64 {
	out := make(map[string]float64, len(counts)+1)
	total := 0
	for _, count := range counts {
		total += count
	}
	if total == 0 {
		out[TopologyGenericHTML] = 1
		return out
	}
	classCount := len(counts)
	denom := float64(total + classCount)
	for cls, count := range counts {
		out[cls] = float64(count+1) / denom
	}
	return out
}

func dominantDistributionKey(distribution map[string]float64, fallback string) string {
	best := fallback
	bestVal := -1.0
	keys := make([]string, 0, len(distribution))
	for key := range distribution {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		val := distribution[key]
		if val > bestVal {
			best = key
			bestVal = val
		}
	}
	return best
}

func dominantCountKey(counts map[string]int, fallback string) string {
	best := fallback
	bestVal := -1
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if counts[key] > bestVal {
			best = key
			bestVal = counts[key]
		}
	}
	return best
}

func fingerprintConfidence(observations int, distribution map[string]float64, robots RobotsAnalysis, health *fetchHealthAccumulator) float64 {
	obsScore := math.Min(1, math.Log(float64(observations+1))/math.Log(50))
	entropyPenalty := float64(distributionEntropyMilli(distribution)) / 1000.0
	healthScore := 0.5
	if health != nil && health.health.Total > 0 {
		healthScore = float64(health.health.Success+1) / float64(health.health.Total+2)
	}
	robotsScore := 0.7
	if robots.HasRobots {
		robotsScore = 1.0
	}
	conf := 0.45*obsScore + 0.35*healthScore + 0.20*robotsScore - 0.15*entropyPenalty
	return clampFloat(conf, 0.05, 1.0)
}

func phaseRecommendation(confidence float64, observations int, health *fetchHealthAccumulator) string {
	successRate := 0.0
	if health != nil && health.health.Total > 0 {
		successRate = float64(health.health.Success) / float64(health.health.Total)
	}
	switch {
	case confidence >= 0.75 && observations >= 50 && successRate >= 0.70:
		return PhaseRecommendationKnown
	case confidence >= 0.40 && observations >= 10:
		return PhaseRecommendationLearning
	default:
		return PhaseRecommendationCold
	}
}

func signalDensityFromPatterns(patterns []URLPattern, pathCount int) float64 {
	if pathCount <= 0 {
		return 0
	}
	score := 0.0
	for _, pattern := range patterns {
		classWeight := 0.5
		switch pattern.TopologyClass {
		case TopologyRESTAPIJSON, TopologySaaSDocs, TopologyNewsArticle, TopologyJSONLDStructured:
			classWeight = 1.0
		case TopologyAuthWall:
			classWeight = 0.1
		}
		score += float64(pattern.Count) * classWeight * clampFloat(pattern.Confidence, 0.1, 1.0)
	}
	return clampFloat(score/float64(pathCount), 0, 1)
}

func estimateFreshnessHalfLife(latestFetchUnix int64, nowUnix int64) float64 {
	if latestFetchUnix <= 0 || nowUnix <= latestFetchUnix {
		return 30
	}
	ageDays := float64(nowUnix-latestFetchUnix) / 86400.0
	switch {
	case ageDays < 1:
		return 3
	case ageDays < 7:
		return 7
	case ageDays < 30:
		return 14
	default:
		return 30
	}
}

func normalizeLanguage(contentLanguage string, headers map[string]string) string {
	candidates := []string{contentLanguage}
	for key, value := range headers {
		if strings.EqualFold(key, "content-language") {
			candidates = append(candidates, value)
		}
	}
	for _, candidate := range candidates {
		candidate = strings.TrimSpace(strings.ToLower(candidate))
		if candidate == "" {
			continue
		}
		candidate = strings.Split(candidate, ",")[0]
		candidate = strings.Split(candidate, ";")[0]
		candidate = strings.TrimSpace(candidate)
		if len(candidate) >= 2 {
			return candidate
		}
	}
	return ""
}

func dominantLanguage(counts map[string]int) string {
	return dominantCountKey(counts, "")
}

func topologyClassIndex(cls string) byte {
	switch cls {
	case TopologyRESTAPIJSON:
		return 1
	case TopologySaaSDocs:
		return 2
	case TopologyEcommerceProduct:
		return 3
	case TopologyNewsArticle:
		return 4
	case TopologyJSONLDStructured:
		return 5
	case TopologySearchResultsPage:
		return 6
	case TopologyMediaGallery:
		return 7
	case TopologyAuthWall:
		return 8
	default:
		return 0
	}
}

func phaseRecommendationIndex(phase string) byte {
	switch phase {
	case PhaseRecommendationKnown:
		return 2
	case PhaseRecommendationLearning:
		return 1
	default:
		return 0
	}
}

func frictionRulesToPatterns(rules []RobotsRule) []PathPattern {
	out := make([]PathPattern, 0, len(rules))
	for _, rule := range rules {
		out = append(out, PathPattern{Pattern: rule.Path, TopologyClass: inferTopology(rule.Path)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Pattern < out[j].Pattern })
	return out
}

func renderRequirementsFromPatterns(patterns []URLPattern, robots RobotsAnalysis) map[string]string {
	out := make(map[string]string, len(patterns))
	for _, pattern := range patterns {
		render := "static"
		if pattern.TopologyClass == TopologyAuthWall || robots.RequiresClearance {
			render = "headless"
		}
		if pattern.TopologyClass == TopologyRESTAPIJSON {
			render = "static"
		}
		out[pattern.Pattern] = render
	}
	return out
}

func distributionEntropyMilli(distribution map[string]float64) int {
	entropy := 0.0
	for _, p := range distribution {
		if p <= 0 {
			continue
		}
		entropy -= p * math.Log2(p)
	}
	return int(math.Round(entropy * 1000))
}

func nonNegativeInt64(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}

func nonNegativeFloat(v float64) float64 {
	if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return v
}

func hashString(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func clampInt(v int, lo int, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampFloat(v float64, lo float64, hi float64) float64 {
	if math.IsNaN(v) {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func comparePatternSets(previous []URLPattern, current []URLPattern) []PatternDelta {
	prevMap := make(map[string]URLPattern, len(previous))
	currMap := make(map[string]URLPattern, len(current))
	for _, pattern := range previous {
		prevMap[pattern.Pattern] = pattern
	}
	for _, pattern := range current {
		currMap[pattern.Pattern] = pattern
	}
	keys := make([]string, 0, len(prevMap)+len(currMap))
	seen := make(map[string]bool)
	for key := range prevMap {
		keys = append(keys, key)
		seen[key] = true
	}
	for key := range currMap {
		if !seen[key] {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	deltas := make([]PatternDelta, 0, len(keys))
	for _, key := range keys {
		prev := prevMap[key]
		curr := currMap[key]
		status := "changed"
		switch {
		case prev.Pattern == "":
			status = "added"
		case curr.Pattern == "":
			status = "removed"
		case prev.Count == curr.Count:
			status = "stable"
		}
		change := 0.0
		if prev.Count == 0 && curr.Count > 0 {
			change = 1
		} else if prev.Count > 0 {
			change = float64(curr.Count-prev.Count) / float64(prev.Count)
		}
		deltas = append(deltas, PatternDelta{
			Pattern:       key,
			PreviousCount: prev.Count,
			CurrentCount:  curr.Count,
			ChangeRatio:   change,
			Status:        status,
		})
	}
	sort.Slice(deltas, func(i, j int) bool {
		ai := math.Abs(deltas[i].ChangeRatio)
		aj := math.Abs(deltas[j].ChangeRatio)
		if ai != aj {
			return ai > aj
		}
		return deltas[i].Pattern < deltas[j].Pattern
	})
	return deltas
}

func compareTopologyDistribution(previous map[string]float64, current map[string]float64) []TopologyDelta {
	keys := make([]string, 0, len(previous)+len(current))
	seen := make(map[string]bool)
	for key := range previous {
		keys = append(keys, key)
		seen[key] = true
	}
	for key := range current {
		if !seen[key] {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	out := make([]TopologyDelta, 0, len(keys))
	for _, key := range keys {
		prev := previous[key]
		curr := current[key]
		out = append(out, TopologyDelta{TopologyClass: key, PreviousShare: prev, CurrentShare: curr, Delta: curr - prev})
	}
	sort.Slice(out, func(i, j int) bool {
		ai := math.Abs(out[i].Delta)
		aj := math.Abs(out[j].Delta)
		if ai != aj {
			return ai > aj
		}
		return out[i].TopologyClass < out[j].TopologyClass
	})
	return out
}

func driftScore(patternDeltas []PatternDelta, topologyDeltas []TopologyDelta, frictionDelta int, confidenceDelta float64, signalDensityDelta float64) float64 {
	patternScore := 0.0
	if len(patternDeltas) > 0 {
		limit := len(patternDeltas)
		if limit > 10 {
			limit = 10
		}
		for i := 0; i < limit; i++ {
			delta := patternDeltas[i]
			weight := 1.0
			if delta.Status == "added" || delta.Status == "removed" {
				weight = 1.5
			}
			patternScore += math.Min(1, math.Abs(delta.ChangeRatio)) * weight
		}
		patternScore = patternScore / float64(limit)
	}
	topologyScore := 0.0
	if len(topologyDeltas) > 0 {
		for _, delta := range topologyDeltas {
			topologyScore += math.Abs(delta.Delta)
		}
		topologyScore = clampFloat(topologyScore/2.0, 0, 1)
	}
	frictionScore := clampFloat(math.Abs(float64(frictionDelta))/3.0, 0, 1)
	confScore := clampFloat(math.Abs(confidenceDelta), 0, 1)
	signalScore := clampFloat(math.Abs(signalDensityDelta), 0, 1)
	return clampFloat(0.35*patternScore+0.30*topologyScore+0.15*frictionScore+0.10*confScore+0.10*signalScore, 0, 1)
}

func driftLevel(score float64) string {
	switch {
	case score >= 0.75:
		return "critical"
	case score >= 0.45:
		return "high"
	case score >= 0.20:
		return "medium"
	case score > 0:
		return "low"
	default:
		return "none"
	}
}

type pathStats struct {
	Patterns     []string
	EntropyMilli int
}

func analyzePaths(paths []string) pathStats {
	counts := map[string]int{}
	total := 0
	for _, raw := range paths {
		p := normalizePath(raw)
		if p == "" {
			continue
		}
		pattern := patternForPath(p)
		counts[pattern]++
		total++
	}
	patterns := make([]string, 0, len(counts))
	for p := range counts {
		patterns = append(patterns, p)
	}
	sort.Strings(patterns)
	entropy := 0
	if total > 0 {
		for _, c := range counts {
			// Integer approximation good enough for deterministic planning.
			probMilli := c * 1000 / total
			entropy += probMilli * (1000 - probMilli) / 1000
		}
	}
	return pathStats{Patterns: patterns, EntropyMilli: entropy}
}

func normalizeDomain(raw string) string {
	raw = strings.TrimSpace(strings.ToLower(raw))
	raw = strings.TrimPrefix(raw, "http://")
	raw = strings.TrimPrefix(raw, "https://")
	raw = strings.TrimSuffix(raw, "/")
	if strings.Contains(raw, "/") {
		raw = strings.Split(raw, "/")[0]
	}
	return raw
}

func normalizePath(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if u, err := url.Parse(raw); err == nil && u.Path != "" {
		raw = u.Path
	}
	if !strings.HasPrefix(raw, "/") {
		raw = "/" + raw
	}
	return raw
}

func patternForPath(p string) string {
	parts := strings.Split(strings.Trim(p, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		return "/"
	}
	if len(parts) == 1 {
		return "/" + parts[0] + "/*"
	}
	return "/" + parts[0] + "/" + parts[1] + "/*"
}

func inferTopology(pattern string) string {
	p := strings.ToLower(pattern)
	switch {
	case strings.Contains(p, "api"):
		return TopologyRESTAPIJSON
	case strings.Contains(p, "docs") || strings.Contains(p, "guide"):
		return TopologySaaSDocs
	case strings.Contains(p, "product") || strings.Contains(p, "shop"):
		return TopologyEcommerceProduct
	case strings.Contains(p, "blog") || strings.Contains(p, "news") || strings.Contains(p, "article"):
		return TopologyNewsArticle
	default:
		return TopologyGenericHTML
	}
}

func parseCrawlDelay(robots string) float64 {
	for _, line := range strings.Split(robots, "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), ":", 2)
		if len(parts) != 2 || !strings.EqualFold(strings.TrimSpace(parts[0]), "crawl-delay") {
			continue
		}
		v := strings.TrimSpace(parts[1])
		if parsed, err := strconv.ParseFloat(v, 64); err == nil && parsed >= 0 {
			return parsed
		}
	}
	return 0
}

func requestsPerSecond(delay float64) float64 {
	if delay <= 0 {
		return 2.0
	}
	if delay < 1 {
		return 1.0
	}
	return 1.0 / delay
}

func burstForDelay(delay float64) int {
	if delay <= 0 {
		return 8
	}
	if delay < 1 {
		return 4
	}
	return 1
}

func buildSeedManifest(domain string, sitemapURLs []string, rate RateLimitProfile, runID string) CrawlManifest {
	urls := make([]CrawlURL, 0, len(sitemapURLs)+2)
	base := "https://" + domain
	urls = append(urls, CrawlURL{URL: base + "/robots.txt", TopologyHint: TopologyGenericHTML, FetchMode: "static", RenderMode: "static", Priority: 0, RateLimitProfile: rate, ExpectedContentType: "text/plain", MaxResponseBytes: 1048576, IsRobots: true, RunID: runID})
	for i, sm := range sitemapURLs {
		urls = append(urls, CrawlURL{URL: sm, TopologyHint: TopologyGenericHTML, FetchMode: "static", RenderMode: "static", Priority: i + 1, RateLimitProfile: rate, ExpectedContentType: "application/xml", MaxResponseBytes: 4194304, IsSitemap: true, RunID: runID})
	}
	return CrawlManifest{Domain: domain, URLs: urls, TotalURLs: len(urls), EstimatedDurationSeconds: float64(len(urls)) / rate.RequestsPerSecond, ClearanceRequired: 1, ManifestID: deterministicID(domain + runID)}
}

func deterministicID(seed string) string {
	sum := sha256.Sum256([]byte(seed))
	return hex.EncodeToString(sum[:16])
}

func buildFingerprint(domain string, paths []string, robots string, sitemap []string) [FingerprintSlotBytes]byte {
	var fp [FingerprintSlotBytes]byte
	h := fnv.New64a()
	h.Write([]byte(domain))
	h.Write([]byte{0})
	h.Write([]byte(robots))
	for _, p := range paths {
		h.Write([]byte{0})
		h.Write([]byte(p))
	}
	for _, s := range sitemap {
		h.Write([]byte{0})
		h.Write([]byte(s))
	}
	binary.LittleEndian.PutUint64(fp[0:8], h.Sum64())
	binary.LittleEndian.PutUint32(fp[8:12], uint32(len(paths)))
	binary.LittleEndian.PutUint32(fp[12:16], uint32(len(sitemap)))
	copy(fp[32:], []byte(domain))
	return fp
}

func fingerprintHash(fp [FingerprintSlotBytes]byte) string {
	sum := sha256.Sum256(fp[:])
	return hex.EncodeToString(sum[:])
}

func disallowedFromRobots(robots string) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(robots, "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), ":", 2)
		if len(parts) == 2 && strings.EqualFold(strings.TrimSpace(parts[0]), "disallow") {
			p := normalizePath(parts[1])
			if p != "" {
				out[p] = inferTopology(p)
			}
		}
	}
	return out
}

func frictionFromRobots(robots string) []PathPattern {
	var out []PathPattern
	for p, tc := range disallowedFromRobots(robots) {
		out = append(out, PathPattern{Pattern: p, TopologyClass: tc})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Pattern < out[j].Pattern })
	return out
}

func detectBotMitigation(robots string) string {
	lower := strings.ToLower(robots)
	if strings.Contains(lower, "cloudflare") {
		return "cloudflare"
	}
	if strings.Contains(lower, "akamai") {
		return "custom"
	}
	return "none"
}

func renderRequirements(patterns []string) map[string]string {
	out := map[string]string{}
	for _, p := range patterns {
		if strings.Contains(strings.ToLower(p), "app") {
			out[p] = "headless"
		} else {
			out[p] = "static"
		}
	}
	return out
}
