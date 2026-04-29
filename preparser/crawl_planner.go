package preparser

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	DefaultPlanMaxURLs        = 128
	DefaultFreshnessLambda    = 0.10
	DefaultPlanLookaheadPaths = 32
	MaxFrontierConcurrency    = 16
	MinFrontierConcurrency    = 1
)

type CrawlPlanInput struct {
	Domain           string
	CandidateURLs    []string
	DomainMap        DomainMap
	PhaseWeight      float64
	FreshnessDecay   float64
	FrictionForecast float64
	RunID            string
}

type PlanOptions struct {
	MaxURLs             int                `json:"max_urls"`
	Phase               string             `json:"phase"`
	DaysSinceLastCrawl  float64            `json:"days_since_last_crawl"`
	FreshnessLambda     float64            `json:"freshness_lambda"`
	NowUnix             int64              `json:"now_unix"`
	MaxConcurrency      int                `json:"max_concurrency"`
	IncludeRobots       bool               `json:"include_robots"`
	IncludeSitemaps     bool               `json:"include_sitemaps"`
	QueryHints          []string           `json:"query_hints"`
	SeenURLHashes       map[string]bool    `json:"seen_url_hashes"`
	TopologyWeights     map[string]float64 `json:"topology_weights"`
	AllowHighFriction   bool               `json:"allow_high_friction"`
	ResumeAfterURL      string             `json:"resume_after_url"`
	ExpectedSignalFloor float64            `json:"expected_signal_floor"`
	PreferFreshPatterns bool               `json:"prefer_fresh_patterns"`
	FrontierComponent   string             `json:"frontier_component"`
}

type PlannedURL struct {
	URL                 string  `json:"url"`
	Path                string  `json:"path"`
	TopologyClass       string  `json:"topology_class"`
	FetchMode           string  `json:"fetch_mode"`
	RenderMode          string  `json:"render_mode"`
	PriorityScore       float64 `json:"priority_score"`
	SignalExpectation   float64 `json:"signal_expectation"`
	FrictionCost        float64 `json:"friction_cost"`
	CrawlDelayMS        int64   `json:"crawl_delay_ms"`
	MaxResponseBytes    int     `json:"max_response_bytes"`
	ExpectedContentType string  `json:"expected_content_type"`
	Reason              string  `json:"reason"`
	ResumeOrdinal       int     `json:"resume_ordinal"`
}

type FrontierPlan struct {
	Domain              string       `json:"domain"`
	Priority            float64      `json:"priority"`
	URLQueue            []PlannedURL `json:"url_queue"`
	RateLimitMS         int64        `json:"rate_limit_ms"`
	MaxConcurrency      int          `json:"max_concurrency"`
	FrictionLevel       int          `json:"friction_level"`
	ResumeToken         string       `json:"resume_token"`
	EstimatedSignal     float64      `json:"estimated_signal"`
	PlanGeneratedAtUnix int64        `json:"plan_generated_at_unix"`
	RunID               string       `json:"run_id"`
	Reason              string       `json:"reason"`
}

type CrawlPlan struct {
	Domain        string        `json:"domain"`
	Manifest      CrawlManifest `json:"manifest"`
	PriorityScore float64       `json:"priority_score"`
	Reason        string        `json:"reason"`
}

type CrawlManifestReadyEvent struct {
	Domain   string        `json:"domain"`
	Manifest CrawlManifest `json:"manifest"`
}

func GeneratePlan(fp *DomainFingerprint, opts PlanOptions) (*FrontierPlan, error) {
	if fp == nil {
		return nil, errors.New("fingerprint is nil")
	}
	if normalizeDomain(fp.Domain) == "" {
		return nil, errors.New("fingerprint domain is empty")
	}
	if opts.MaxURLs <= 0 {
		opts.MaxURLs = DefaultPlanMaxURLs
	}
	if opts.FreshnessLambda <= 0 {
		opts.FreshnessLambda = DefaultFreshnessLambda
	}
	if opts.NowUnix <= 0 {
		opts.NowUnix = time.Now().Unix()
	}
	if opts.Phase == "" {
		opts.Phase = fp.PhaseRecommendation
	}
	if opts.Phase == "" {
		opts.Phase = PhaseRecommendationCold
	}
	runID := fp.RunID
	if runID == "" {
		runID = deterministicID(fp.Domain + strconvUnix(opts.NowUnix))
	}
	candidates := CandidateQueueFromFingerprint(fp, opts)
	if len(candidates) == 0 {
		candidates = fallbackCandidates(fp, opts)
	}
	planned := make([]PlannedURL, 0, len(candidates))
	phase := PhaseWeight(opts.Phase)
	freshness := FreshnessDecay(opts.DaysSinceLastCrawl, opts.FreshnessLambda)
	friction := frictionCostForLevel(fp.FrictionLevel)
	for _, candidate := range candidates {
		if opts.SeenURLHashes != nil && opts.SeenURLHashes[hashURL(candidate.URL)] {
			continue
		}
		if candidate.SignalExpectation < opts.ExpectedSignalFloor {
			continue
		}
		candidate.PriorityScore = priorityScore(candidate.SignalExpectation, phase, freshness, friction)
		candidate.FrictionCost = friction
		candidate.CrawlDelayMS = RateLimitDelayMS(fp.RobotsSignals.CrawlDelaySeconds, fp.FrictionLevel)
		candidate.FetchMode = plannedFetchMode(candidate.RenderMode, fp.FrictionLevel, opts.AllowHighFriction)
		candidate.ExpectedContentType = expectedContentType(candidate.TopologyClass)
		if candidate.MaxResponseBytes == 0 {
			candidate.MaxResponseBytes = maxBytesForTopology(candidate.TopologyClass)
		}
		planned = append(planned, candidate)
	}
	sortPlannedURLs(planned)
	planned = trimPlannedAfterResume(planned, opts.ResumeAfterURL)
	if len(planned) > opts.MaxURLs {
		planned = planned[:opts.MaxURLs]
	}
	for i := range planned {
		planned[i].ResumeOrdinal = i
	}
	estimatedSignal := estimatePlanSignal(planned)
	plan := &FrontierPlan{
		Domain:              fp.Domain,
		Priority:            clampFloat(priorityScore(fp.SignalDensity, phase, freshness, friction), 0, 1),
		URLQueue:            planned,
		RateLimitMS:         RateLimitDelayMS(fp.RobotsSignals.CrawlDelaySeconds, fp.FrictionLevel),
		MaxConcurrency:      planConcurrency(fp.FrictionLevel, opts.MaxConcurrency),
		FrictionLevel:       fp.FrictionLevel,
		EstimatedSignal:     estimatedSignal,
		PlanGeneratedAtUnix: opts.NowUnix,
		RunID:               runID,
		Reason:              planReason(fp, opts, len(planned)),
	}
	plan.ResumeToken = BuildResumeToken(plan)
	return plan, nil
}

func PrioritizePlan(plans []*FrontierPlan) []*FrontierPlan {
	out := make([]*FrontierPlan, 0, len(plans))
	for _, plan := range plans {
		if plan != nil {
			out = append(out, plan)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Priority != out[j].Priority {
			return out[i].Priority > out[j].Priority
		}
		if out[i].EstimatedSignal != out[j].EstimatedSignal {
			return out[i].EstimatedSignal > out[j].EstimatedSignal
		}
		return out[i].Domain < out[j].Domain
	})
	return out
}

func SerializePlan(plan *FrontierPlan) ([]byte, error) {
	if plan == nil {
		return nil, errors.New("plan is nil")
	}
	return json.Marshal(plan)
}

func DeserializePlan(data []byte) (*FrontierPlan, error) {
	if len(data) == 0 {
		return nil, errors.New("plan data is empty")
	}
	var plan FrontierPlan
	if err := json.Unmarshal(data, &plan); err != nil {
		return nil, err
	}
	if plan.Domain == "" {
		return nil, errors.New("plan domain is empty")
	}
	return &plan, nil
}

func BuildResumeToken(plan *FrontierPlan) string {
	if plan == nil {
		return ""
	}
	var buf [40]byte
	sum := sha256.Sum256([]byte(plan.Domain + plan.RunID + strconvUnix(plan.PlanGeneratedAtUnix)))
	copy(buf[0:16], sum[:16])
	binary.LittleEndian.PutUint64(buf[16:24], uint64(plan.PlanGeneratedAtUnix))
	binary.LittleEndian.PutUint32(buf[24:28], uint32(len(plan.URLQueue)))
	binary.LittleEndian.PutUint32(buf[28:32], uint32(clampInt(plan.FrictionLevel, FrictionLevelCL1, FrictionLevelCL4)))
	copy(buf[32:40], []byte(plan.Domain))
	return base64.RawURLEncoding.EncodeToString(buf[:])
}

func DecodeResumeToken(token string) ([]byte, error) {
	if token == "" {
		return nil, errors.New("resume token is empty")
	}
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil {
		return nil, err
	}
	if len(raw) != 40 {
		return nil, errors.New("resume token has invalid length")
	}
	return raw, nil
}

func PlanCrawl(input CrawlPlanInput) (CrawlPlan, error) {
	domain := normalizeDomain(input.Domain)
	if domain == "" {
		return CrawlPlan{}, errors.New("domain is empty")
	}
	if input.RunID == "" {
		return CrawlPlan{}, errors.New("run_id is empty")
	}
	rate := input.DomainMap.RateLimitProfile
	if rate.Domain == "" {
		rate = RateLimitProfile{Domain: domain, RequestsPerSecond: 2, BurstCapacity: 8}
	}
	urls := uniqueURLs(domain, input.CandidateURLs)
	crawlURLs := make([]CrawlURL, 0, len(urls))
	for i, u := range urls {
		path := pathOf(u)
		topology := topologyForPath(input.DomainMap, path)
		render := renderForPath(input.DomainMap, path)
		fetch := fetchModeFor(render, input.FrictionForecast)
		crawlURLs = append(crawlURLs, CrawlURL{
			URL:                 u,
			TopologyHint:        topology,
			FetchMode:           fetch,
			RenderMode:          render,
			Priority:            i,
			RateLimitProfile:    rate,
			ExpectedContentType: expectedContentType(topology),
			CrawlDelaySeconds:   rate.CrawlDelaySeconds,
			MaxResponseBytes:    4 * 1024 * 1024,
			RunID:               input.RunID,
		})
	}
	score := priorityScore(signalDensity(input.DomainMap), input.PhaseWeight, input.FreshnessDecay, frictionCost(input.FrictionForecast))
	manifest := CrawlManifest{Domain: domain, URLs: crawlURLs, TotalURLs: len(crawlURLs), EstimatedDurationSeconds: estimateDuration(crawlURLs, rate), ClearanceRequired: clearance(crawlURLs), ManifestID: deterministicID(domain + input.RunID + "plan")}
	return CrawlPlan{Domain: domain, Manifest: manifest, PriorityScore: score, Reason: "signal_density_phase_freshness_over_friction"}, nil
}

func (p CrawlPlan) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "crawl_manifest", Component: "preparser.crawl_planner", Payload: CrawlManifestReadyEvent{Domain: p.Domain, Manifest: p.Manifest}}
}

func (p FrontierPlan) BridgeEvent() BridgeRequest {
	manifest := CrawlManifest{
		Domain:                   p.Domain,
		URLs:                     crawlURLsFromPlanned(p.Domain, p.URLQueue, p.RunID),
		TotalURLs:                len(p.URLQueue),
		EstimatedDurationSeconds: estimateFrontierDurationSeconds(p),
		ClearanceRequired:        p.FrictionLevel,
		ManifestID:               deterministicID(p.Domain + p.RunID + p.ResumeToken),
	}
	return BridgeRequest{Topic: "crawl_manifest", Component: "preparser.crawl_planner", Payload: CrawlManifestReadyEvent{Domain: p.Domain, Manifest: manifest}}
}

func CandidateQueueFromFingerprint(fp *DomainFingerprint, opts PlanOptions) []PlannedURL {
	if fp == nil {
		return nil
	}
	patterns := append([]URLPattern(nil), fp.URLPatterns...)
	sort.SliceStable(patterns, func(i, j int) bool {
		if patterns[i].Confidence != patterns[j].Confidence {
			return patterns[i].Confidence > patterns[j].Confidence
		}
		if patterns[i].Count != patterns[j].Count {
			return patterns[i].Count > patterns[j].Count
		}
		return patterns[i].Pattern < patterns[j].Pattern
	})
	limit := DefaultPlanLookaheadPaths
	if opts.MaxURLs > 0 && opts.MaxURLs < limit {
		limit = opts.MaxURLs
	}
	out := make([]PlannedURL, 0, limit+len(fp.RobotsSignals.SitemapURLs))
	if opts.IncludeRobots {
		out = append(out, PlannedURL{
			URL:                 "https://" + fp.Domain + "/robots.txt",
			Path:                "/robots.txt",
			TopologyClass:       TopologyGenericHTML,
			RenderMode:          "static",
			SignalExpectation:   0.1,
			MaxResponseBytes:    1024 * 1024,
			ExpectedContentType: "text/plain",
			Reason:              "robots_refresh",
		})
	}
	if opts.IncludeSitemaps {
		for _, sitemap := range fp.RobotsSignals.SitemapURLs {
			out = append(out, PlannedURL{
				URL:                 sitemap,
				Path:                pathOf(sitemap),
				TopologyClass:       TopologyGenericHTML,
				RenderMode:          "static",
				SignalExpectation:   0.25,
				MaxResponseBytes:    4 * 1024 * 1024,
				ExpectedContentType: "application/xml",
				Reason:              "sitemap_refresh",
			})
		}
	}
	for _, pattern := range patterns {
		if len(out) >= limit {
			break
		}
		if blockedByRobots(pattern.Pattern, fp.RobotsSignals) && !opts.AllowHighFriction {
			continue
		}
		urls := expandPatternCandidates(fp.Domain, pattern, opts)
		for _, candidateURL := range urls {
			if len(out) >= limit {
				break
			}
			topologyWeight := topologyWeight(pattern.TopologyClass, opts.TopologyWeights)
			expectation := clampFloat(pattern.Confidence*topologyWeight*classSignalPrior(pattern.TopologyClass), 0, 1)
			out = append(out, PlannedURL{
				URL:                 candidateURL,
				Path:                pathOf(candidateURL),
				TopologyClass:       pattern.TopologyClass,
				RenderMode:          renderModeForPattern(pattern, fp.RobotsSignals),
				SignalExpectation:   expectation,
				MaxResponseBytes:    maxBytesForTopology(pattern.TopologyClass),
				ExpectedContentType: expectedContentType(pattern.TopologyClass),
				Reason:              "pattern:" + pattern.Pattern,
			})
		}
	}
	return dedupePlannedURLs(out)
}

func fallbackCandidates(fp *DomainFingerprint, opts PlanOptions) []PlannedURL {
	if fp == nil {
		return nil
	}
	paths := []string{"/", "/robots.txt"}
	if len(opts.QueryHints) > 0 {
		for _, hint := range opts.QueryHints {
			hint = strings.TrimSpace(strings.ToLower(hint))
			if hint == "" {
				continue
			}
			hint = strings.ReplaceAll(hint, " ", "-")
			paths = append(paths, "/"+hint)
		}
	}
	out := make([]PlannedURL, 0, len(paths))
	for _, path := range paths {
		topology := inferTopology(path)
		out = append(out, PlannedURL{
			URL:                 "https://" + fp.Domain + normalizePath(path),
			Path:                normalizePath(path),
			TopologyClass:       topology,
			RenderMode:          "static",
			SignalExpectation:   classSignalPrior(topology) * 0.4,
			MaxResponseBytes:    maxBytesForTopology(topology),
			ExpectedContentType: expectedContentType(topology),
			Reason:              "fallback_seed",
		})
	}
	return out
}

func expandPatternCandidates(domain string, pattern URLPattern, opts PlanOptions) []string {
	base := "https://" + domain
	clean := strings.TrimSuffix(pattern.Pattern, "/*")
	clean = strings.TrimSuffix(clean, "/")
	if clean == "" {
		clean = "/"
	}
	candidates := make([]string, 0, 4)
	if !strings.Contains(clean, "*") {
		candidates = append(candidates, base+normalizePath(clean))
	} else {
		replacements := replacementTokensForPattern(pattern, opts)
		for _, repl := range replacements {
			path := strings.Replace(clean, "*", repl, 1)
			if strings.Contains(path, "*") {
				path = strings.ReplaceAll(path, "*", "index")
			}
			candidates = append(candidates, base+normalizePath(path))
		}
	}
	for _, example := range pattern.Examples {
		if strings.HasPrefix(example, "http://") || strings.HasPrefix(example, "https://") {
			candidates = append(candidates, example)
		} else {
			candidates = append(candidates, base+normalizePath(example))
		}
	}
	sort.Strings(candidates)
	return uniqueURLs(domain, candidates)
}

func replacementTokensForPattern(pattern URLPattern, opts PlanOptions) []string {
	tokens := make([]string, 0, 4)
	for _, hint := range opts.QueryHints {
		hint = strings.TrimSpace(strings.ToLower(hint))
		if hint == "" {
			continue
		}
		tokens = append(tokens, strings.ReplaceAll(hint, " ", "-"))
	}
	if len(tokens) == 0 {
		switch pattern.TopologyClass {
		case TopologySaaSDocs:
			tokens = append(tokens, "guide", "api", "overview")
		case TopologyEcommerceProduct:
			tokens = append(tokens, "featured", "latest", "index")
		case TopologyNewsArticle:
			tokens = append(tokens, "latest", "index", "archive")
		case TopologyRESTAPIJSON:
			tokens = append(tokens, "v1", "index", "status")
		default:
			tokens = append(tokens, "index", "latest", "overview")
		}
	}
	return tokens
}

func blockedByRobots(pattern string, robots RobotsAnalysis) bool {
	for _, rule := range robots.DisallowRules {
		if rule.Path == "/" {
			return true
		}
		if strings.HasPrefix(pattern, strings.TrimSuffix(rule.Path, "*")) {
			return true
		}
	}
	return false
}

func renderModeForPattern(pattern URLPattern, robots RobotsAnalysis) string {
	if robots.RequiresClearance || pattern.TopologyClass == TopologyAuthWall {
		return "headless"
	}
	if strings.Contains(strings.ToLower(pattern.Pattern), "app") {
		return "headless"
	}
	return "static"
}

func topologyWeight(topology string, weights map[string]float64) float64 {
	if weights != nil {
		if weight, ok := weights[topology]; ok && weight > 0 {
			return weight
		}
	}
	return 1
}

func classSignalPrior(topology string) float64 {
	switch topology {
	case TopologyRESTAPIJSON:
		return 0.95
	case TopologySaaSDocs:
		return 0.90
	case TopologyNewsArticle:
		return 0.85
	case TopologyJSONLDStructured:
		return 0.80
	case TopologyEcommerceProduct:
		return 0.70
	case TopologySearchResultsPage:
		return 0.55
	case TopologyMediaGallery:
		return 0.35
	case TopologyAuthWall:
		return 0.10
	default:
		return 0.45
	}
}

func maxBytesForTopology(topology string) int {
	switch topology {
	case TopologyRESTAPIJSON, TopologyJSONLDStructured:
		return 2 * 1024 * 1024
	case TopologyMediaGallery:
		return 256 * 1024
	case TopologySaaSDocs, TopologyNewsArticle:
		return 6 * 1024 * 1024
	default:
		return 4 * 1024 * 1024
	}
}

func dedupePlannedURLs(in []PlannedURL) []PlannedURL {
	seen := make(map[string]bool, len(in))
	out := make([]PlannedURL, 0, len(in))
	for _, item := range in {
		if item.URL == "" || seen[item.URL] {
			continue
		}
		seen[item.URL] = true
		out = append(out, item)
	}
	return out
}

func sortPlannedURLs(urls []PlannedURL) {
	sort.SliceStable(urls, func(i, j int) bool {
		if urls[i].PriorityScore != urls[j].PriorityScore {
			return urls[i].PriorityScore > urls[j].PriorityScore
		}
		if urls[i].SignalExpectation != urls[j].SignalExpectation {
			return urls[i].SignalExpectation > urls[j].SignalExpectation
		}
		if urls[i].FrictionCost != urls[j].FrictionCost {
			return urls[i].FrictionCost < urls[j].FrictionCost
		}
		return urls[i].URL < urls[j].URL
	})
}

func trimPlannedAfterResume(urls []PlannedURL, resumeAfterURL string) []PlannedURL {
	if resumeAfterURL == "" {
		return urls
	}
	for i, item := range urls {
		if item.URL == resumeAfterURL {
			return append([]PlannedURL(nil), urls[i+1:]...)
		}
	}
	return urls
}

func estimatePlanSignal(urls []PlannedURL) float64 {
	if len(urls) == 0 {
		return 0
	}
	total := 0.0
	for _, item := range urls {
		total += item.SignalExpectation
	}
	return total / float64(len(urls))
}

func RateLimitDelayMS(crawlDelaySeconds float64, frictionLevel int) int64 {
	delay := crawlDelaySeconds * 1000
	switch frictionLevel {
	case FrictionLevelCL4:
		delay = math.Max(delay, 5000)
	case FrictionLevelCL3:
		delay = math.Max(delay, 2000)
	case FrictionLevelCL2:
		delay = math.Max(delay, 500)
	default:
		delay = math.Max(delay, 100)
	}
	return int64(delay)
}

func planConcurrency(frictionLevel int, requested int) int {
	maxForFriction := MaxConcurrencyForFriction(frictionLevel)
	if requested <= 0 || requested > maxForFriction {
		return maxForFriction
	}
	return clampInt(requested, MinFrontierConcurrency, maxForFriction)
}

func MaxConcurrencyForFriction(frictionLevel int) int {
	switch frictionLevel {
	case FrictionLevelCL4:
		return 1
	case FrictionLevelCL3:
		return 2
	case FrictionLevelCL2:
		return 4
	default:
		return MaxFrontierConcurrency
	}
}

func PhaseWeight(phase string) float64 {
	switch phase {
	case PhaseRecommendationKnown:
		return 1.0
	case PhaseRecommendationLearning:
		return 0.6
	default:
		return 0.3
	}
}

func FreshnessDecay(daysSinceLastCrawl float64, lambda float64) float64 {
	if lambda <= 0 {
		lambda = DefaultFreshnessLambda
	}
	if daysSinceLastCrawl < 0 {
		daysSinceLastCrawl = 0
	}
	return math.Exp(-lambda * daysSinceLastCrawl)
}

func frictionCostForLevel(level int) float64 {
	switch level {
	case FrictionLevelCL4:
		return 4.0
	case FrictionLevelCL3:
		return 2.5
	case FrictionLevelCL2:
		return 1.5
	default:
		return 1.0
	}
}

func plannedFetchMode(renderMode string, frictionLevel int, allowHighFriction bool) string {
	if frictionLevel >= FrictionLevelCL4 && allowHighFriction {
		return "tor_full"
	}
	if frictionLevel >= FrictionLevelCL3 {
		return "tor"
	}
	if renderMode == "headless" || frictionLevel == FrictionLevelCL2 {
		return "headless"
	}
	return "static"
}

func planReason(fp *DomainFingerprint, opts PlanOptions, count int) string {
	parts := []string{
		"phase=" + opts.Phase,
		"friction=CL" + strconv.Itoa(fp.FrictionLevel),
		"urls=" + strconv.Itoa(count),
		"density=" + strconv.FormatFloat(fp.SignalDensity, 'f', 3, 64),
	}
	if opts.ResumeAfterURL != "" {
		parts = append(parts, "resumed=true")
	}
	if opts.PreferFreshPatterns {
		parts = append(parts, "fresh_patterns=true")
	}
	return strings.Join(parts, " ")
}

func crawlURLsFromPlanned(domain string, in []PlannedURL, runID string) []CrawlURL {
	out := make([]CrawlURL, 0, len(in))
	for i, item := range in {
		out = append(out, CrawlURL{
			URL:                 item.URL,
			TopologyHint:        item.TopologyClass,
			FetchMode:           item.FetchMode,
			RenderMode:          item.RenderMode,
			Priority:            i,
			RateLimitProfile:    RateLimitProfile{Domain: domain, RequestsPerSecond: requestsPerSecond(float64(item.CrawlDelayMS) / 1000.0), CrawlDelaySeconds: float64(item.CrawlDelayMS) / 1000.0, BurstCapacity: burstForDelay(float64(item.CrawlDelayMS) / 1000.0)},
			ExpectedContentType: item.ExpectedContentType,
			CrawlDelaySeconds:   float64(item.CrawlDelayMS) / 1000.0,
			MaxResponseBytes:    item.MaxResponseBytes,
			RunID:               runID,
		})
	}
	return out
}

func estimateFrontierDurationSeconds(plan FrontierPlan) float64 {
	if len(plan.URLQueue) == 0 {
		return 0
	}
	concurrency := plan.MaxConcurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	delay := float64(plan.RateLimitMS) / 1000.0
	return float64(len(plan.URLQueue)) * delay / float64(concurrency)
}

func hashURL(raw string) string {
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}

func strconvUnix(v int64) string {
	return strconv.FormatInt(v, 10)
}

func uniqueURLs(domain string, in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, raw := range in {
		u := strings.TrimSpace(raw)
		if u == "" {
			continue
		}
		if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
			u = "https://" + domain + normalizePath(u)
		}
		if parsed, err := url.Parse(u); err == nil && normalizeDomain(parsed.Host) == domain && !seen[u] {
			seen[u] = true
			out = append(out, u)
		}
	}
	sort.Strings(out)
	return out
}

func pathOf(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return normalizePath(raw)
	}
	return normalizePath(u.Path)
}

func topologyForPath(dm DomainMap, path string) string {
	best := ""
	bestLen := -1
	for pattern, tc := range dm.PathTopologyMap {
		prefix := strings.TrimSuffix(pattern, "*")
		if strings.HasPrefix(path, prefix) && len(prefix) > bestLen {
			best = tc
			bestLen = len(prefix)
		}
	}
	if best != "" {
		return best
	}
	return inferTopology(path)
}

func renderForPath(dm DomainMap, path string) string {
	for pattern, render := range dm.RenderRequirements {
		prefix := strings.TrimSuffix(pattern, "*")
		if strings.HasPrefix(path, prefix) && render != "" {
			return render
		}
	}
	return "static"
}

func fetchModeFor(render string, friction float64) string {
	if friction >= 0.75 {
		return "tor"
	}
	if render == "headless" || friction >= 0.35 {
		return "headless"
	}
	return "static"
}

func expectedContentType(topology string) string {
	if topology == "REST_API_JSON" || topology == "JSON_LD_STRUCTURED" {
		return "application/json"
	}
	return "text/html"
}

func signalDensity(dm DomainMap) float64 {
	if dm.ObservedPathCount == 0 {
		return 0.1
	}
	return float64(len(dm.SignalZones)+1) / float64(dm.ObservedPathCount+1)
}

func frictionCost(f float64) float64 {
	if f < 0 {
		f = 0
	}
	return 1.0 + f
}

func priorityScore(signal, phase, freshness, friction float64) float64 {
	if phase <= 0 {
		phase = 1
	}
	if freshness <= 0 {
		freshness = 1
	}
	if friction <= 0 {
		friction = 1
	}
	return (signal * phase * freshness) / friction
}

func estimateDuration(urls []CrawlURL, rate RateLimitProfile) float64 {
	rps := rate.RequestsPerSecond
	if rps <= 0 {
		rps = 1
	}
	return float64(len(urls)) / rps
}

func clearance(urls []CrawlURL) int {
	level := 1
	for _, u := range urls {
		switch u.FetchMode {
		case "tor_full":
			return 4
		case "tor":
			if level < 3 {
				level = 3
			}
		case "headless":
			if level < 2 {
				level = 2
			}
		}
	}
	return level
}

// ─── Frontier Scheduling ────────────────────────────────────────────────────

// FrontierScheduler manages ordered dispatch of frontier plans across domains.
// It maintains a priority heap of pending plans and enforces global concurrency
// limits to prevent overloading the fetch layer.
type FrontierScheduler struct {
	mu          sync.RWMutex
	plans       []*FrontierPlan
	dispatched  map[string]int64
	maxActive   int
	activeCount int
}

// FrontierSchedulerConfig holds tunable parameters for the scheduler.
type FrontierSchedulerConfig struct {
	MaxActiveDomains    int     `json:"max_active_domains"`
	MinPlanPriority     float64 `json:"min_plan_priority"`
	StarvationLimitSecs int64   `json:"starvation_limit_secs"`
	RebalanceInterval   int64   `json:"rebalance_interval"`
}

// NewFrontierScheduler creates a scheduler with the given concurrency cap.
func NewFrontierScheduler(maxActive int) *FrontierScheduler {
	if maxActive <= 0 {
		maxActive = 8
	}
	return &FrontierScheduler{
		plans:      make([]*FrontierPlan, 0, 32),
		dispatched: make(map[string]int64),
		maxActive:  maxActive,
	}
}

// Enqueue adds a plan to the scheduler, maintaining priority order.
func (s *FrontierScheduler) Enqueue(plan *FrontierPlan) error {
	if plan == nil {
		return errors.New("plan is nil")
	}
	if plan.Domain == "" {
		return errors.New("plan domain is empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, existing := range s.plans {
		if existing.Domain == plan.Domain {
			s.plans[i] = plan
			s.sortPlansLocked()
			return nil
		}
	}
	s.plans = append(s.plans, plan)
	s.sortPlansLocked()
	return nil
}

// Dequeue returns the highest priority plan that can be dispatched, or nil.
func (s *FrontierScheduler) Dequeue() *FrontierPlan {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.activeCount >= s.maxActive || len(s.plans) == 0 {
		return nil
	}
	plan := s.plans[0]
	s.plans = s.plans[1:]
	s.activeCount++
	s.dispatched[plan.Domain] = time.Now().Unix()
	return plan
}

// Complete marks a domain's plan as finished, freeing a concurrency slot.
func (s *FrontierScheduler) Complete(domain string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	domain = normalizeDomain(domain)
	if _, ok := s.dispatched[domain]; ok {
		delete(s.dispatched, domain)
		if s.activeCount > 0 {
			s.activeCount--
		}
	}
}

// PendingCount returns the number of plans waiting for dispatch.
func (s *FrontierScheduler) PendingCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.plans)
}

// ActiveCount returns the number of plans currently being executed.
func (s *FrontierScheduler) ActiveCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.activeCount
}

// PendingDomains returns a sorted list of domains with pending plans.
func (s *FrontierScheduler) PendingDomains() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.plans))
	for _, p := range s.plans {
		out = append(out, p.Domain)
	}
	return out
}

// ActiveDomains returns a sorted list of currently dispatched domains.
func (s *FrontierScheduler) ActiveDomains() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.dispatched))
	for d := range s.dispatched {
		out = append(out, d)
	}
	sort.Strings(out)
	return out
}

// DrainStale removes plans that have been dispatched longer than limitSecs.
func (s *FrontierScheduler) DrainStale(limitSecs int64) []string {
	if limitSecs <= 0 {
		limitSecs = 300
	}
	now := time.Now().Unix()
	s.mu.Lock()
	defer s.mu.Unlock()
	var drained []string
	for domain, ts := range s.dispatched {
		if now-ts > limitSecs {
			delete(s.dispatched, domain)
			if s.activeCount > 0 {
				s.activeCount--
			}
			drained = append(drained, domain)
		}
	}
	sort.Strings(drained)
	return drained
}

func (s *FrontierScheduler) sortPlansLocked() {
	sort.SliceStable(s.plans, func(i, j int) bool {
		if s.plans[i].Priority != s.plans[j].Priority {
			return s.plans[i].Priority > s.plans[j].Priority
		}
		if s.plans[i].EstimatedSignal != s.plans[j].EstimatedSignal {
			return s.plans[i].EstimatedSignal > s.plans[j].EstimatedSignal
		}
		return s.plans[i].Domain < s.plans[j].Domain
	})
}

// ─── Plan Validation ────────────────────────────────────────────────────────

// PlanValidationIssue represents a single problem found in a frontier plan.
type PlanValidationIssue struct {
	Field    string `json:"field"`
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
}

// PlanValidationReport contains the result of validating a frontier plan.
type PlanValidationReport struct {
	Valid      bool                  `json:"valid"`
	Domain     string                `json:"domain"`
	Issues     []PlanValidationIssue `json:"issues"`
	ErrorCount int                   `json:"error_count"`
	WarnCount  int                   `json:"warn_count"`
	URLCount   int                   `json:"url_count"`
}

// ValidateFrontierPlan checks a plan for structural and semantic issues.
func ValidateFrontierPlan(plan *FrontierPlan) PlanValidationReport {
	report := PlanValidationReport{Valid: true}
	if plan == nil {
		report.Valid = false
		report.Issues = append(report.Issues, PlanValidationIssue{
			Field: "plan", Code: "nil_plan", Severity: "error", Message: "plan is nil",
		})
		report.ErrorCount++
		return report
	}
	report.Domain = plan.Domain
	report.URLCount = len(plan.URLQueue)
	addErr := func(field, code, msg string) {
		report.Issues = append(report.Issues, PlanValidationIssue{Field: field, Code: code, Severity: "error", Message: msg})
		report.ErrorCount++
		report.Valid = false
	}
	addWarn := func(field, code, msg string) {
		report.Issues = append(report.Issues, PlanValidationIssue{Field: field, Code: code, Severity: "warning", Message: msg})
		report.WarnCount++
	}
	if normalizeDomain(plan.Domain) == "" {
		addErr("domain", "empty_domain", "plan domain is empty")
	}
	if plan.RunID == "" {
		addErr("run_id", "empty_run_id", "plan run_id is empty")
	}
	if plan.Priority < 0 || plan.Priority > 1 {
		addWarn("priority", "priority_range", "priority outside 0..1")
	}
	if plan.MaxConcurrency <= 0 {
		addWarn("max_concurrency", "zero_concurrency", "max concurrency is zero or negative")
	}
	if plan.RateLimitMS <= 0 {
		addWarn("rate_limit_ms", "zero_rate_limit", "rate limit is zero or negative")
	}
	if plan.FrictionLevel < FrictionLevelCL1 || plan.FrictionLevel > FrictionLevelCL4 {
		addErr("friction_level", "invalid_friction", "friction level outside CL1-CL4")
	}
	if len(plan.URLQueue) == 0 {
		addWarn("url_queue", "empty_queue", "plan has no URLs")
	}
	seen := make(map[string]bool, len(plan.URLQueue))
	for i, u := range plan.URLQueue {
		if u.URL == "" {
			addErr("url_queue", "empty_url", "URL at index "+strconv.Itoa(i)+" is empty")
			continue
		}
		if seen[u.URL] {
			addWarn("url_queue", "duplicate_url", "duplicate URL: "+u.URL)
		}
		seen[u.URL] = true
		if u.TopologyClass == "" {
			addWarn("url_queue", "missing_topology", "URL "+u.URL+" has no topology class")
		}
		if u.SignalExpectation < 0 || u.SignalExpectation > 1 {
			addWarn("url_queue", "signal_range", "signal expectation outside 0..1 for "+u.URL)
		}
	}
	if plan.ResumeToken == "" {
		addWarn("resume_token", "missing_token", "plan has no resume token")
	}
	return report
}

// ─── Adaptive Plan Merging ──────────────────────────────────────────────────

// MergePlans combines two frontier plans for the same domain, keeping the
// highest-priority URLs and deduplicating. The newer plan's metadata wins.
func MergePlans(existing *FrontierPlan, incoming *FrontierPlan) (*FrontierPlan, error) {
	if existing == nil && incoming == nil {
		return nil, errors.New("both plans are nil")
	}
	if existing == nil {
		return incoming, nil
	}
	if incoming == nil {
		return existing, nil
	}
	if normalizeDomain(existing.Domain) != normalizeDomain(incoming.Domain) {
		return nil, errors.New("cannot merge plans for different domains")
	}
	merged := &FrontierPlan{
		Domain:              incoming.Domain,
		Priority:            math.Max(existing.Priority, incoming.Priority),
		RateLimitMS:         maxInt64(existing.RateLimitMS, incoming.RateLimitMS),
		MaxConcurrency:      minIntPositive(existing.MaxConcurrency, incoming.MaxConcurrency),
		FrictionLevel:       maxIntVal(existing.FrictionLevel, incoming.FrictionLevel),
		PlanGeneratedAtUnix: incoming.PlanGeneratedAtUnix,
		RunID:               incoming.RunID,
		Reason:              "merged:" + incoming.Reason,
	}
	seen := make(map[string]bool)
	combined := make([]PlannedURL, 0, len(existing.URLQueue)+len(incoming.URLQueue))
	for _, u := range incoming.URLQueue {
		if u.URL != "" && !seen[u.URL] {
			seen[u.URL] = true
			combined = append(combined, u)
		}
	}
	for _, u := range existing.URLQueue {
		if u.URL != "" && !seen[u.URL] {
			seen[u.URL] = true
			combined = append(combined, u)
		}
	}
	sortPlannedURLs(combined)
	if len(combined) > DefaultPlanMaxURLs*2 {
		combined = combined[:DefaultPlanMaxURLs*2]
	}
	for i := range combined {
		combined[i].ResumeOrdinal = i
	}
	merged.URLQueue = combined
	merged.EstimatedSignal = estimatePlanSignal(combined)
	merged.ResumeToken = BuildResumeToken(merged)
	return merged, nil
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func minIntPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}

func maxIntVal(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ─── Batch Plan Generation ──────────────────────────────────────────────────

// BatchPlanInput groups fingerprints with shared options for batch planning.
type BatchPlanInput struct {
	Fingerprints []*DomainFingerprint
	Options      PlanOptions
}

// BatchPlanResult holds a plan and any error from batch generation.
type BatchPlanResult struct {
	Plan  *FrontierPlan `json:"plan"`
	Error string        `json:"error,omitempty"`
}

// GenerateBatchPlans creates frontier plans for multiple domains at once.
func GenerateBatchPlans(input BatchPlanInput) []BatchPlanResult {
	results := make([]BatchPlanResult, 0, len(input.Fingerprints))
	for _, fp := range input.Fingerprints {
		plan, err := GeneratePlan(fp, input.Options)
		if err != nil {
			results = append(results, BatchPlanResult{Error: err.Error()})
			continue
		}
		results = append(results, BatchPlanResult{Plan: plan})
	}
	return results
}

// GenerateBatchPlansFromStore creates plans by reading fingerprints from a store.
func GenerateBatchPlansFromStore(domains []string, store CursorStore, opts PlanOptions, runID string) ([]BatchPlanResult, error) {
	if store == nil {
		return nil, errors.New("cursor store is nil")
	}
	analyzer := NewDomainAnalyzer()
	fingerprints, err := analyzer.BatchAnalyzeStore(domains, store, runID)
	if err != nil {
		return nil, err
	}
	return GenerateBatchPlans(BatchPlanInput{Fingerprints: fingerprints, Options: opts}), nil
}

// ─── Plan Persistence ───────────────────────────────────────────────────────

// PlanStore defines the interface for persisting frontier plans.
type PlanStore interface {
	SavePlan(plan *FrontierPlan) error
	LoadPlan(domain string) (*FrontierPlan, error)
	ListPlans() ([]string, error)
	DeletePlan(domain string) error
}

// MemoryPlanStore implements PlanStore using an in-memory map.
type MemoryPlanStore struct {
	mu    sync.RWMutex
	plans map[string]*FrontierPlan
}

// NewMemoryPlanStore creates a new in-memory plan store.
func NewMemoryPlanStore() *MemoryPlanStore {
	return &MemoryPlanStore{plans: make(map[string]*FrontierPlan)}
}

func (s *MemoryPlanStore) SavePlan(plan *FrontierPlan) error {
	if s == nil || plan == nil {
		return errors.New("nil store or plan")
	}
	domain := normalizeDomain(plan.Domain)
	if domain == "" {
		return errors.New("plan domain is empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.plans[domain] = plan
	return nil
}

func (s *MemoryPlanStore) LoadPlan(domain string) (*FrontierPlan, error) {
	if s == nil {
		return nil, errors.New("nil store")
	}
	domain = normalizeDomain(domain)
	s.mu.RLock()
	defer s.mu.RUnlock()
	plan, ok := s.plans[domain]
	if !ok {
		return nil, errors.New("plan not found: " + domain)
	}
	return plan, nil
}

func (s *MemoryPlanStore) ListPlans() ([]string, error) {
	if s == nil {
		return nil, errors.New("nil store")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.plans))
	for domain := range s.plans {
		out = append(out, domain)
	}
	sort.Strings(out)
	return out, nil
}

func (s *MemoryPlanStore) DeletePlan(domain string) error {
	if s == nil {
		return errors.New("nil store")
	}
	domain = normalizeDomain(domain)
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.plans, domain)
	return nil
}

// ─── Plan Summary & Diagnostics ─────────────────────────────────────────────

// PlanSummary contains human-readable plan diagnostics.
type PlanSummary struct {
	Domain             string         `json:"domain"`
	URLCount           int            `json:"url_count"`
	Priority           float64        `json:"priority"`
	EstimatedSignal    float64        `json:"estimated_signal"`
	FrictionLevel      int            `json:"friction_level"`
	MaxConcurrency     int            `json:"max_concurrency"`
	RateLimitMS        int64          `json:"rate_limit_ms"`
	TopologyBreakdown  map[string]int `json:"topology_breakdown"`
	FetchModeBreakdown map[string]int `json:"fetch_mode_breakdown"`
	TotalExpectedBytes int64          `json:"total_expected_bytes"`
}

// SummarizePlan generates a diagnostic summary of a frontier plan.
func SummarizePlan(plan *FrontierPlan) PlanSummary {
	summary := PlanSummary{
		TopologyBreakdown:  make(map[string]int),
		FetchModeBreakdown: make(map[string]int),
	}
	if plan == nil {
		return summary
	}
	summary.Domain = plan.Domain
	summary.URLCount = len(plan.URLQueue)
	summary.Priority = plan.Priority
	summary.EstimatedSignal = plan.EstimatedSignal
	summary.FrictionLevel = plan.FrictionLevel
	summary.MaxConcurrency = plan.MaxConcurrency
	summary.RateLimitMS = plan.RateLimitMS
	for _, u := range plan.URLQueue {
		tc := u.TopologyClass
		if tc == "" {
			tc = "unknown"
		}
		summary.TopologyBreakdown[tc]++
		fm := u.FetchMode
		if fm == "" {
			fm = "static"
		}
		summary.FetchModeBreakdown[fm]++
		summary.TotalExpectedBytes += int64(u.MaxResponseBytes)
	}
	return summary
}

// PlanSummaryLine returns a one-line summary string for logging.
func PlanSummaryLine(plan *FrontierPlan) string {
	if plan == nil {
		return "plan=nil"
	}
	return strings.Join([]string{
		"domain=" + plan.Domain,
		"urls=" + strconv.Itoa(len(plan.URLQueue)),
		"priority=" + strconv.FormatFloat(plan.Priority, 'f', 3, 64),
		"signal=" + strconv.FormatFloat(plan.EstimatedSignal, 'f', 3, 64),
		"friction=CL" + strconv.Itoa(plan.FrictionLevel),
		"concurrency=" + strconv.Itoa(plan.MaxConcurrency),
		"rate_ms=" + strconv.FormatInt(plan.RateLimitMS, 10),
	}, " ")
}

// ─── Plan Rebalancing ───────────────────────────────────────────────────────

// RebalanceConfig controls how plans are rebalanced across domains.
type RebalanceConfig struct {
	MaxTotalURLs       int     `json:"max_total_urls"`
	MinURLsPerDomain   int     `json:"min_urls_per_domain"`
	PriorityFloor      float64 `json:"priority_floor"`
	SignalFloor        float64 `json:"signal_floor"`
	MaxDomainsPerCycle int     `json:"max_domains_per_cycle"`
}

// DefaultRebalanceConfig returns sensible defaults for rebalancing.
func DefaultRebalanceConfig() RebalanceConfig {
	return RebalanceConfig{
		MaxTotalURLs:       512,
		MinURLsPerDomain:   4,
		PriorityFloor:      0.05,
		SignalFloor:        0.01,
		MaxDomainsPerCycle: 32,
	}
}

// RebalancePlans adjusts URL counts across plans to fit within global limits.
func RebalancePlans(plans []*FrontierPlan, config RebalanceConfig) []*FrontierPlan {
	if len(plans) == 0 {
		return nil
	}
	if config.MaxTotalURLs <= 0 {
		config.MaxTotalURLs = 512
	}
	if config.MinURLsPerDomain <= 0 {
		config.MinURLsPerDomain = 4
	}
	if config.MaxDomainsPerCycle <= 0 {
		config.MaxDomainsPerCycle = 32
	}
	filtered := make([]*FrontierPlan, 0, len(plans))
	for _, p := range plans {
		if p == nil || p.Priority < config.PriorityFloor {
			continue
		}
		if p.EstimatedSignal < config.SignalFloor && len(p.URLQueue) > 0 {
			continue
		}
		filtered = append(filtered, p)
	}
	if len(filtered) == 0 {
		return nil
	}
	sorted := PrioritizePlan(filtered)
	if len(sorted) > config.MaxDomainsPerCycle {
		sorted = sorted[:config.MaxDomainsPerCycle]
	}
	totalPriority := 0.0
	for _, p := range sorted {
		totalPriority += p.Priority
	}
	if totalPriority <= 0 {
		totalPriority = float64(len(sorted))
	}
	remaining := config.MaxTotalURLs
	out := make([]*FrontierPlan, 0, len(sorted))
	for _, p := range sorted {
		share := int(float64(config.MaxTotalURLs) * (p.Priority / totalPriority))
		if share < config.MinURLsPerDomain {
			share = config.MinURLsPerDomain
		}
		if share > remaining {
			share = remaining
		}
		if share <= 0 {
			continue
		}
		trimmed := trimPlanToSize(p, share)
		out = append(out, trimmed)
		remaining -= len(trimmed.URLQueue)
		if remaining <= 0 {
			break
		}
	}
	return out
}

func trimPlanToSize(plan *FrontierPlan, maxURLs int) *FrontierPlan {
	if plan == nil {
		return nil
	}
	cp := *plan
	if len(cp.URLQueue) > maxURLs {
		cp.URLQueue = append([]PlannedURL(nil), cp.URLQueue[:maxURLs]...)
	} else {
		cp.URLQueue = append([]PlannedURL(nil), cp.URLQueue...)
	}
	for i := range cp.URLQueue {
		cp.URLQueue[i].ResumeOrdinal = i
	}
	cp.EstimatedSignal = estimatePlanSignal(cp.URLQueue)
	cp.ResumeToken = BuildResumeToken(&cp)
	return &cp
}

// ─── Frontier Cursor ────────────────────────────────────────────────────────

// FrontierCursor tracks crawl progress within a plan for resume capability.
type FrontierCursor struct {
	Domain         string `json:"domain"`
	LastURL        string `json:"last_url"`
	LastOrdinal    int    `json:"last_ordinal"`
	CompletedCount int    `json:"completed_count"`
	FailedCount    int    `json:"failed_count"`
	SkippedCount   int    `json:"skipped_count"`
	ResumeToken    string `json:"resume_token"`
	StartedAtUnix  int64  `json:"started_at_unix"`
	UpdatedAtUnix  int64  `json:"updated_at_unix"`
	RunID          string `json:"run_id"`
}

// NewFrontierCursor creates a cursor for a plan.
func NewFrontierCursor(plan *FrontierPlan) FrontierCursor {
	if plan == nil {
		return FrontierCursor{}
	}
	now := time.Now().Unix()
	return FrontierCursor{
		Domain:        plan.Domain,
		ResumeToken:   plan.ResumeToken,
		StartedAtUnix: now,
		UpdatedAtUnix: now,
		RunID:         plan.RunID,
	}
}

// Advance moves the cursor forward after a URL is processed.
func (c *FrontierCursor) Advance(u string, succeeded bool) {
	if c == nil {
		return
	}
	c.LastURL = u
	c.LastOrdinal++
	c.UpdatedAtUnix = time.Now().Unix()
	if succeeded {
		c.CompletedCount++
	} else {
		c.FailedCount++
	}
}

// Skip records a skipped URL.
func (c *FrontierCursor) Skip(u string) {
	if c == nil {
		return
	}
	c.LastURL = u
	c.LastOrdinal++
	c.SkippedCount++
	c.UpdatedAtUnix = time.Now().Unix()
}

// Progress returns the fraction of URLs processed (0..1).
func (c FrontierCursor) Progress(totalURLs int) float64 {
	if totalURLs <= 0 {
		return 0
	}
	processed := c.CompletedCount + c.FailedCount + c.SkippedCount
	return clampFloat(float64(processed)/float64(totalURLs), 0, 1)
}

// SuccessRate returns the fraction of completed URLs that succeeded.
func (c FrontierCursor) SuccessRate() float64 {
	total := c.CompletedCount + c.FailedCount
	if total <= 0 {
		return 0
	}
	return float64(c.CompletedCount) / float64(total)
}

// IsComplete returns true if all URLs have been processed.
func (c FrontierCursor) IsComplete(totalURLs int) bool {
	return c.CompletedCount+c.FailedCount+c.SkippedCount >= totalURLs
}

// ─── Query-Guided Planning ──────────────────────────────────────────────────

// QueryGuidedPlanInput extends plan generation with a user query for relevance.
type QueryGuidedPlanInput struct {
	Query       string
	Fingerprint *DomainFingerprint
	Options     PlanOptions
}

// GenerateQueryGuidedPlan creates a plan biased toward a user's query terms.
func GenerateQueryGuidedPlan(input QueryGuidedPlanInput) (*FrontierPlan, error) {
	if input.Fingerprint == nil {
		return nil, errors.New("fingerprint is nil")
	}
	query := strings.TrimSpace(input.Query)
	if query == "" {
		return GeneratePlan(input.Fingerprint, input.Options)
	}
	opts := input.Options
	terms := extractQueryTerms(query)
	if len(terms) > 0 {
		opts.QueryHints = mergeHints(opts.QueryHints, terms)
	}
	topologyBias := queryTopologyBias(query)
	if topologyBias != "" && opts.TopologyWeights == nil {
		opts.TopologyWeights = make(map[string]float64)
	}
	if topologyBias != "" {
		opts.TopologyWeights[topologyBias] = 2.0
	}
	opts.PreferFreshPatterns = true
	return GeneratePlan(input.Fingerprint, opts)
}

func extractQueryTerms(query string) []string {
	words := strings.Fields(strings.ToLower(query))
	stopwords := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "can": true, "how": true, "what": true,
		"which": true, "who": true, "whom": true, "this": true, "that": true,
		"these": true, "those": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "for": true,
		"of": true, "with": true, "by": true, "from": true, "it": true,
		"its": true, "not": true, "no": true, "so": true, "if": true,
	}
	out := make([]string, 0, len(words))
	for _, w := range words {
		w = strings.Trim(w, ".,;:!?\"'()[]{}")
		if len(w) < 2 || stopwords[w] {
			continue
		}
		out = append(out, w)
	}
	return out
}

func mergeHints(existing []string, terms []string) []string {
	seen := make(map[string]bool, len(existing))
	for _, h := range existing {
		seen[h] = true
	}
	out := append([]string(nil), existing...)
	for _, t := range terms {
		if !seen[t] {
			out = append(out, t)
			seen[t] = true
		}
	}
	return out
}

func queryTopologyBias(query string) string {
	lower := strings.ToLower(query)
	switch {
	case strings.Contains(lower, "api") || strings.Contains(lower, "endpoint"):
		return TopologyRESTAPIJSON
	case strings.Contains(lower, "docs") || strings.Contains(lower, "documentation") || strings.Contains(lower, "guide"):
		return TopologySaaSDocs
	case strings.Contains(lower, "news") || strings.Contains(lower, "article") || strings.Contains(lower, "blog"):
		return TopologyNewsArticle
	case strings.Contains(lower, "product") || strings.Contains(lower, "price") || strings.Contains(lower, "buy"):
		return TopologyEcommerceProduct
	default:
		return ""
	}
}

// ─── Plan Diffing ───────────────────────────────────────────────────────────

// PlanDiff describes changes between two versions of a plan.
type PlanDiff struct {
	Domain        string   `json:"domain"`
	AddedURLs     []string `json:"added_urls"`
	RemovedURLs   []string `json:"removed_urls"`
	RetainedURLs  []string `json:"retained_urls"`
	PriorityDelta float64  `json:"priority_delta"`
	SignalDelta   float64  `json:"signal_delta"`
}

// DiffPlans compares two plans and returns the differences.
func DiffPlans(old *FrontierPlan, new_ *FrontierPlan) PlanDiff {
	diff := PlanDiff{}
	if old == nil && new_ == nil {
		return diff
	}
	if old != nil {
		diff.Domain = old.Domain
	}
	if new_ != nil {
		diff.Domain = new_.Domain
	}
	oldURLs := make(map[string]bool)
	newURLs := make(map[string]bool)
	if old != nil {
		for _, u := range old.URLQueue {
			oldURLs[u.URL] = true
		}
	}
	if new_ != nil {
		for _, u := range new_.URLQueue {
			newURLs[u.URL] = true
		}
	}
	for u := range newURLs {
		if oldURLs[u] {
			diff.RetainedURLs = append(diff.RetainedURLs, u)
		} else {
			diff.AddedURLs = append(diff.AddedURLs, u)
		}
	}
	for u := range oldURLs {
		if !newURLs[u] {
			diff.RemovedURLs = append(diff.RemovedURLs, u)
		}
	}
	sort.Strings(diff.AddedURLs)
	sort.Strings(diff.RemovedURLs)
	sort.Strings(diff.RetainedURLs)
	oldP, newP := 0.0, 0.0
	oldS, newS := 0.0, 0.0
	if old != nil {
		oldP = old.Priority
		oldS = old.EstimatedSignal
	}
	if new_ != nil {
		newP = new_.Priority
		newS = new_.EstimatedSignal
	}
	diff.PriorityDelta = newP - oldP
	diff.SignalDelta = newS - oldS
	return diff
}

// ─── Plan Event Bridge ──────────────────────────────────────────────────────

// PlanReadyEvent is emitted when a plan is ready for dispatch.
type PlanReadyEvent struct {
	Domain          string  `json:"domain"`
	URLCount        int     `json:"url_count"`
	Priority        float64 `json:"priority"`
	EstimatedSignal float64 `json:"estimated_signal"`
	FrictionLevel   int     `json:"friction_level"`
	RunID           string  `json:"run_id"`
}

// PlanCompletedEvent is emitted when a plan finishes execution.
type PlanCompletedEvent struct {
	Domain        string  `json:"domain"`
	CompletedURLs int     `json:"completed_urls"`
	FailedURLs    int     `json:"failed_urls"`
	SkippedURLs   int     `json:"skipped_urls"`
	SuccessRate   float64 `json:"success_rate"`
	DurationSecs  float64 `json:"duration_secs"`
	RunID         string  `json:"run_id"`
}

// PlanStaleEvent is emitted when a plan has not made progress.
type PlanStaleEvent struct {
	Domain      string `json:"domain"`
	StaleSecs   int64  `json:"stale_secs"`
	LastOrdinal int    `json:"last_ordinal"`
	RunID       string `json:"run_id"`
}

func (e PlanReadyEvent) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "plan_ready", Component: "preparser.crawl_planner", Payload: e}
}

func (e PlanCompletedEvent) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "plan_completed", Component: "preparser.crawl_planner", Payload: e}
}

func (e PlanStaleEvent) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "plan_stale", Component: "preparser.crawl_planner", Payload: e}
}

// BuildPlanReadyEvent constructs a ready event from a plan.
func BuildPlanReadyEvent(plan *FrontierPlan) PlanReadyEvent {
	if plan == nil {
		return PlanReadyEvent{}
	}
	return PlanReadyEvent{
		Domain:          plan.Domain,
		URLCount:        len(plan.URLQueue),
		Priority:        plan.Priority,
		EstimatedSignal: plan.EstimatedSignal,
		FrictionLevel:   plan.FrictionLevel,
		RunID:           plan.RunID,
	}
}

// BuildPlanCompletedEvent constructs a completed event from a cursor.
func BuildPlanCompletedEvent(cursor FrontierCursor) PlanCompletedEvent {
	duration := 0.0
	if cursor.UpdatedAtUnix > cursor.StartedAtUnix {
		duration = float64(cursor.UpdatedAtUnix - cursor.StartedAtUnix)
	}
	return PlanCompletedEvent{
		Domain:        cursor.Domain,
		CompletedURLs: cursor.CompletedCount,
		FailedURLs:    cursor.FailedCount,
		SkippedURLs:   cursor.SkippedCount,
		SuccessRate:   cursor.SuccessRate(),
		DurationSecs:  duration,
		RunID:         cursor.RunID,
	}
}

// ─── Plan Statistics ────────────────────────────────────────────────────────

// PlanTopologyStats provides signal density analysis by topology class.
type PlanTopologyStats struct {
	TopologyClass      string  `json:"topology_class"`
	URLCount           int     `json:"url_count"`
	MeanSignal         float64 `json:"mean_signal"`
	MeanFriction       float64 `json:"mean_friction"`
	TotalExpectedBytes int64   `json:"total_expected_bytes"`
}

// ComputePlanTopologyStats breaks down a plan by topology class.
func ComputePlanTopologyStats(plan *FrontierPlan) []PlanTopologyStats {
	if plan == nil || len(plan.URLQueue) == 0 {
		return nil
	}
	type acc struct {
		count       int
		signalSum   float64
		frictionSum float64
		bytesSum    int64
	}
	buckets := make(map[string]*acc)
	for _, u := range plan.URLQueue {
		tc := u.TopologyClass
		if tc == "" {
			tc = "unknown"
		}
		a, ok := buckets[tc]
		if !ok {
			a = &acc{}
			buckets[tc] = a
		}
		a.count++
		a.signalSum += u.SignalExpectation
		a.frictionSum += u.FrictionCost
		a.bytesSum += int64(u.MaxResponseBytes)
	}
	keys := make([]string, 0, len(buckets))
	for k := range buckets {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]PlanTopologyStats, 0, len(keys))
	for _, k := range keys {
		a := buckets[k]
		out = append(out, PlanTopologyStats{
			TopologyClass:      k,
			URLCount:           a.count,
			MeanSignal:         a.signalSum / float64(a.count),
			MeanFriction:       a.frictionSum / float64(a.count),
			TotalExpectedBytes: a.bytesSum,
		})
	}
	return out
}

// ─── Plan Filtering ─────────────────────────────────────────────────────────

// FilterPlanURLs removes URLs from a plan that don't meet criteria.
func FilterPlanURLs(plan *FrontierPlan, minSignal float64, excludeTopologies []string) *FrontierPlan {
	if plan == nil {
		return nil
	}
	excludeSet := make(map[string]bool, len(excludeTopologies))
	for _, t := range excludeTopologies {
		excludeSet[t] = true
	}
	cp := *plan
	filtered := make([]PlannedURL, 0, len(cp.URLQueue))
	for _, u := range cp.URLQueue {
		if u.SignalExpectation < minSignal {
			continue
		}
		if excludeSet[u.TopologyClass] {
			continue
		}
		filtered = append(filtered, u)
	}
	for i := range filtered {
		filtered[i].ResumeOrdinal = i
	}
	cp.URLQueue = filtered
	cp.EstimatedSignal = estimatePlanSignal(filtered)
	cp.ResumeToken = BuildResumeToken(&cp)
	return &cp
}

// ─── Plan Serialization Helpers ─────────────────────────────────────────────

// SerializePlanSummary returns a compact JSON summary of a plan.
func SerializePlanSummary(plan *FrontierPlan) ([]byte, error) {
	if plan == nil {
		return nil, errors.New("plan is nil")
	}
	summary := SummarizePlan(plan)
	return json.Marshal(summary)
}

// SerializePlanDiff returns JSON for a plan diff.
func SerializePlanDiff(diff PlanDiff) ([]byte, error) {
	return json.Marshal(diff)
}

// SerializeBatchResults returns JSON for batch plan results.
func SerializeBatchResults(results []BatchPlanResult) ([]byte, error) {
	return json.Marshal(results)
}

// SerializePlanValidation returns JSON for a validation report.
func SerializePlanValidation(report PlanValidationReport) ([]byte, error) {
	return json.Marshal(report)
}

// ─── URL Priority Adjustment ────────────────────────────────────────────────

// AdjustURLPriorities recalculates priority scores with updated parameters.
func AdjustURLPriorities(plan *FrontierPlan, phaseWeight float64, freshnessDecay float64) *FrontierPlan {
	if plan == nil {
		return nil
	}
	cp := *plan
	cp.URLQueue = append([]PlannedURL(nil), cp.URLQueue...)
	friction := frictionCostForLevel(cp.FrictionLevel)
	if phaseWeight <= 0 {
		phaseWeight = PhaseWeight(PhaseRecommendationCold)
	}
	if freshnessDecay <= 0 {
		freshnessDecay = 1.0
	}
	for i := range cp.URLQueue {
		cp.URLQueue[i].PriorityScore = priorityScore(
			cp.URLQueue[i].SignalExpectation,
			phaseWeight,
			freshnessDecay,
			friction,
		)
		cp.URLQueue[i].FrictionCost = friction
	}
	sortPlannedURLs(cp.URLQueue)
	for i := range cp.URLQueue {
		cp.URLQueue[i].ResumeOrdinal = i
	}
	cp.Priority = clampFloat(
		priorityScore(cp.EstimatedSignal, phaseWeight, freshnessDecay, friction), 0, 1,
	)
	cp.ResumeToken = BuildResumeToken(&cp)
	return &cp
}

// ─── Frontier Health Check ──────────────────────────────────────────────────

// FrontierHealthReport summarizes scheduler health.
type FrontierHealthReport struct {
	PendingPlans   int      `json:"pending_plans"`
	ActivePlans    int      `json:"active_plans"`
	PendingDomains []string `json:"pending_domains"`
	ActiveDomains  []string `json:"active_domains"`
	StaleDomains   []string `json:"stale_domains"`
}

// CheckFrontierHealth produces a health report from the scheduler.
func CheckFrontierHealth(scheduler *FrontierScheduler, staleLimitSecs int64) FrontierHealthReport {
	if scheduler == nil {
		return FrontierHealthReport{}
	}
	stale := scheduler.DrainStale(staleLimitSecs)
	return FrontierHealthReport{
		PendingPlans:   scheduler.PendingCount(),
		ActivePlans:    scheduler.ActiveCount(),
		PendingDomains: scheduler.PendingDomains(),
		ActiveDomains:  scheduler.ActiveDomains(),
		StaleDomains:   stale,
	}
}

// ─── Plan Execution Tracking ────────────────────────────────────────────────

// ExecutionRecord captures the outcome of fetching a single planned URL.
type ExecutionRecord struct {
	URL            string  `json:"url"`
	TopologyClass  string  `json:"topology_class"`
	StatusCode     int     `json:"status_code"`
	ResponseBytes  int64   `json:"response_bytes"`
	LatencyMS      float64 `json:"latency_ms"`
	Succeeded      bool    `json:"succeeded"`
	SignalBytes    int     `json:"signal_bytes"`
	SignalDensity  float64 `json:"signal_density"`
	FetchMode      string  `json:"fetch_mode"`
	ExecutedAtUnix int64   `json:"executed_at_unix"`
	RunID          string  `json:"run_id"`
}

// ExecutionBatch groups execution records for a domain plan.
type ExecutionBatch struct {
	Domain  string            `json:"domain"`
	Records []ExecutionRecord `json:"records"`
	Cursor  FrontierCursor    `json:"cursor"`
	RunID   string            `json:"run_id"`
}

// ExecutionBatchSummary summarizes an execution batch for bus events.
type ExecutionBatchSummary struct {
	Domain           string  `json:"domain"`
	TotalURLs        int     `json:"total_urls"`
	SucceededURLs    int     `json:"succeeded_urls"`
	FailedURLs       int     `json:"failed_urls"`
	TotalBytes       int64   `json:"total_bytes"`
	TotalSignalBytes int64   `json:"total_signal_bytes"`
	MeanLatencyMS    float64 `json:"mean_latency_ms"`
	MeanDensity      float64 `json:"mean_density"`
	RunID            string  `json:"run_id"`
}

// SummarizeExecution produces an aggregate summary from an execution batch.
func SummarizeExecution(batch ExecutionBatch) ExecutionBatchSummary {
	summary := ExecutionBatchSummary{
		Domain: batch.Domain,
		RunID:  batch.RunID,
	}
	if len(batch.Records) == 0 {
		return summary
	}
	latencySum := 0.0
	densitySum := 0.0
	for _, r := range batch.Records {
		summary.TotalURLs++
		if r.Succeeded {
			summary.SucceededURLs++
		} else {
			summary.FailedURLs++
		}
		summary.TotalBytes += r.ResponseBytes
		summary.TotalSignalBytes += int64(r.SignalBytes)
		latencySum += r.LatencyMS
		densitySum += r.SignalDensity
	}
	summary.MeanLatencyMS = latencySum / float64(summary.TotalURLs)
	summary.MeanDensity = densitySum / float64(summary.TotalURLs)
	return summary
}

func (s ExecutionBatchSummary) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "execution_batch", Component: "preparser.crawl_planner", Payload: s}
}

// ─── Adaptive Rate Limiting ─────────────────────────────────────────────────

// AdaptiveRateLimiter adjusts crawl delays based on server response patterns.
type AdaptiveRateLimiter struct {
	mu             sync.Mutex
	domain         string
	baseDelayMS    int64
	currentDelayMS int64
	successStreak  int
	failureStreak  int
	maxDelayMS     int64
	minDelayMS     int64
}

// NewAdaptiveRateLimiter creates a limiter with the given base delay.
func NewAdaptiveRateLimiter(domain string, baseDelayMS int64) *AdaptiveRateLimiter {
	if baseDelayMS <= 0 {
		baseDelayMS = 500
	}
	return &AdaptiveRateLimiter{
		domain:         normalizeDomain(domain),
		baseDelayMS:    baseDelayMS,
		currentDelayMS: baseDelayMS,
		maxDelayMS:     baseDelayMS * 20,
		minDelayMS:     baseDelayMS / 4,
	}
}

// RecordSuccess decreases delay after consecutive successes.
func (l *AdaptiveRateLimiter) RecordSuccess() {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.failureStreak = 0
	l.successStreak++
	if l.successStreak >= 5 && l.currentDelayMS > l.minDelayMS {
		l.currentDelayMS = l.currentDelayMS * 3 / 4
		if l.currentDelayMS < l.minDelayMS {
			l.currentDelayMS = l.minDelayMS
		}
		l.successStreak = 0
	}
}

// RecordFailure increases delay after consecutive failures.
func (l *AdaptiveRateLimiter) RecordFailure(statusCode int) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.successStreak = 0
	l.failureStreak++
	multiplier := int64(2)
	if statusCode == 429 {
		multiplier = 4
	}
	l.currentDelayMS = l.currentDelayMS * multiplier
	if l.currentDelayMS > l.maxDelayMS {
		l.currentDelayMS = l.maxDelayMS
	}
}

// CurrentDelayMS returns the current rate limit delay.
func (l *AdaptiveRateLimiter) CurrentDelayMS() int64 {
	if l == nil {
		return 500
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.currentDelayMS
}

// Reset restores the limiter to its base delay.
func (l *AdaptiveRateLimiter) Reset() {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.currentDelayMS = l.baseDelayMS
	l.successStreak = 0
	l.failureStreak = 0
}

// ─── Plan Replay ────────────────────────────────────────────────────────────

// ReplayPlan rebuilds a plan's URL queue from a cursor position, skipping
// already-processed URLs.
func ReplayPlan(plan *FrontierPlan, cursor FrontierCursor) *FrontierPlan {
	if plan == nil {
		return nil
	}
	cp := *plan
	if cursor.LastOrdinal <= 0 || cursor.LastOrdinal >= len(cp.URLQueue) {
		return &cp
	}
	cp.URLQueue = append([]PlannedURL(nil), cp.URLQueue[cursor.LastOrdinal:]...)
	for i := range cp.URLQueue {
		cp.URLQueue[i].ResumeOrdinal = i
	}
	cp.EstimatedSignal = estimatePlanSignal(cp.URLQueue)
	cp.ResumeToken = BuildResumeToken(&cp)
	return &cp
}

// ─── Plan Cost Estimation ───────────────────────────────────────────────────

// PlanCostEstimate provides resource usage projections for a plan.
type PlanCostEstimate struct {
	Domain                 string  `json:"domain"`
	EstimatedDurationSecs  float64 `json:"estimated_duration_secs"`
	EstimatedBandwidthMB   float64 `json:"estimated_bandwidth_mb"`
	EstimatedRequests      int     `json:"estimated_requests"`
	ConcurrencySlots       int     `json:"concurrency_slots"`
	FrictionCostMultiplier float64 `json:"friction_cost_multiplier"`
}

// EstimatePlanCost projects resource usage for a plan.
func EstimatePlanCost(plan *FrontierPlan) PlanCostEstimate {
	if plan == nil {
		return PlanCostEstimate{}
	}
	totalBytes := int64(0)
	for _, u := range plan.URLQueue {
		totalBytes += int64(u.MaxResponseBytes)
	}
	concurrency := plan.MaxConcurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	delayPerReq := float64(plan.RateLimitMS) / 1000.0
	duration := float64(len(plan.URLQueue)) * delayPerReq / float64(concurrency)
	return PlanCostEstimate{
		Domain:                 plan.Domain,
		EstimatedDurationSecs:  duration,
		EstimatedBandwidthMB:   float64(totalBytes) / (1024 * 1024),
		EstimatedRequests:      len(plan.URLQueue),
		ConcurrencySlots:       concurrency,
		FrictionCostMultiplier: frictionCostForLevel(plan.FrictionLevel),
	}
}

// ─── URL Scoring Utilities ──────────────────────────────────────────────────

// ScoreURL computes a composite score for a single URL candidate.
func ScoreURL(u PlannedURL, phaseWeight float64, freshnessDecay float64) float64 {
	friction := 1.0
	if u.FrictionCost > 0 {
		friction = u.FrictionCost
	}
	if phaseWeight <= 0 {
		phaseWeight = 0.3
	}
	if freshnessDecay <= 0 {
		freshnessDecay = 1.0
	}
	return priorityScore(u.SignalExpectation, phaseWeight, freshnessDecay, friction)
}

// TopNURLs returns the top N URLs from a plan by priority score.
func TopNURLs(plan *FrontierPlan, n int) []PlannedURL {
	if plan == nil || n <= 0 {
		return nil
	}
	sorted := append([]PlannedURL(nil), plan.URLQueue...)
	sortPlannedURLs(sorted)
	if len(sorted) > n {
		sorted = sorted[:n]
	}
	return sorted
}

// ─── Plan Wire Format ───────────────────────────────────────────────────────

// EncodePlanForWire serializes a plan for transmission over the bus socket.
func EncodePlanForWire(plan *FrontierPlan) ([]byte, error) {
	if plan == nil {
		return nil, errors.New("plan is nil")
	}
	event := plan.BridgeEvent()
	return json.Marshal(event)
}

// DecodePlanFromWire deserializes a plan from a bus socket message.
func DecodePlanFromWire(data []byte) (*CrawlManifestReadyEvent, error) {
	if len(data) == 0 {
		return nil, errors.New("empty wire data")
	}
	var bridge BridgeRequest
	if err := json.Unmarshal(data, &bridge); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(bridge.Payload)
	if err != nil {
		return nil, err
	}
	var event CrawlManifestReadyEvent
	if err := json.Unmarshal(payload, &event); err != nil {
		return nil, err
	}
	if event.Domain == "" {
		return nil, errors.New("decoded event has empty domain")
	}
	return &event, nil
}

// ─── Plan Optimization & Packing ────────────────────────────────────────────

// KnapsackPacker optimizes URL selection to maximize signal within a bandwidth budget.
type KnapsackPacker struct {
	MaxBandwidthBytes int64
	MaxRequests       int
}

// Pack URLs using a greedy heuristic that approximates the 0/1 knapsack problem
// for signal density (SignalBytes / MaxResponseBytes).
func (p KnapsackPacker) Pack(urls []PlannedURL) []PlannedURL {
	if len(urls) == 0 {
		return nil
	}
	if p.MaxBandwidthBytes <= 0 && p.MaxRequests <= 0 {
		return append([]PlannedURL(nil), urls...)
	}

	// Create a copy to sort by value density
	candidates := append([]PlannedURL(nil), urls...)
	sort.SliceStable(candidates, func(i, j int) bool {
		di := float64(candidates[i].SignalExpectation) / float64(candidates[i].MaxResponseBytes+1)
		dj := float64(candidates[j].SignalExpectation) / float64(candidates[j].MaxResponseBytes+1)
		if math.Abs(di-dj) > 0.0001 {
			return di > dj
		}
		// Tie-break by highest absolute signal
		return candidates[i].SignalExpectation > candidates[j].SignalExpectation
	})

	var packed []PlannedURL
	var currentBytes int64
	var currentReqs int

	for _, u := range candidates {
		if p.MaxRequests > 0 && currentReqs >= p.MaxRequests {
			break
		}
		if p.MaxBandwidthBytes > 0 && currentBytes+int64(u.MaxResponseBytes) > p.MaxBandwidthBytes {
			continue
		}
		packed = append(packed, u)
		currentBytes += int64(u.MaxResponseBytes)
		currentReqs++
	}

	// Restore original ordering (usually by priority) for the packed set
	sortPlannedURLs(packed)
	return packed
}

// OptimizePlan applies the packing algorithm to a plan, ensuring it fits within budgets.
func OptimizePlan(plan *FrontierPlan, maxBytes int64, maxReqs int) *FrontierPlan {
	if plan == nil {
		return nil
	}
	packer := KnapsackPacker{
		MaxBandwidthBytes: maxBytes,
		MaxRequests:       maxReqs,
	}
	cp := *plan
	cp.URLQueue = packer.Pack(cp.URLQueue)
	for i := range cp.URLQueue {
		cp.URLQueue[i].ResumeOrdinal = i
	}
	cp.EstimatedSignal = estimatePlanSignal(cp.URLQueue)
	cp.ResumeToken = BuildResumeToken(&cp)
	return &cp
}

// ─── Health-Based Plan Throttling ───────────────────────────────────────────

// PlanThrottle controls dynamic pausing and backoff during plan execution based
// on real-time error rates.
type PlanThrottle struct {
	mu                  sync.Mutex
	errorThresholdRatio float64
	minObservations     int
	backoffDurationMS   int64
	pausedUntilUnix     int64
	consecutiveErrors   int
}

// NewPlanThrottle creates a throttle that triggers when the error ratio exceeds the threshold.
func NewPlanThrottle(thresholdRatio float64, minObs int, backoffMS int64) *PlanThrottle {
	if thresholdRatio <= 0 {
		thresholdRatio = 0.25
	}
	if minObs <= 0 {
		minObs = 10
	}
	if backoffMS <= 0 {
		backoffMS = 30000
	}
	return &PlanThrottle{
		errorThresholdRatio: thresholdRatio,
		minObservations:     minObs,
		backoffDurationMS:   backoffMS,
	}
}

// Observe incorporates the result of a fetch and determines if a pause is needed.
func (pt *PlanThrottle) Observe(succeeded bool, totalAttempted int, totalFailed int) {
	if pt == nil {
		return
	}
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if !succeeded {
		pt.consecutiveErrors++
	} else {
		pt.consecutiveErrors = 0
	}

	// Immediate backoff on hard failure streaks
	if pt.consecutiveErrors >= 5 {
		pt.pausedUntilUnix = time.Now().Unix() + (pt.backoffDurationMS / 1000)
		return
	}

	if totalAttempted >= pt.minObservations {
		ratio := float64(totalFailed) / float64(totalAttempted)
		if ratio >= pt.errorThresholdRatio {
			pt.pausedUntilUnix = time.Now().Unix() + (pt.backoffDurationMS / 1000)
		}
	}
}

// IsPaused returns true if the plan is currently under a backoff period.
func (pt *PlanThrottle) IsPaused() bool {
	if pt == nil {
		return false
	}
	pt.mu.Lock()
	defer pt.mu.Unlock()
	return time.Now().Unix() < pt.pausedUntilUnix
}

// PausedUntil returns the unix timestamp when the plan can resume.
func (pt *PlanThrottle) PausedUntil() int64 {
	if pt == nil {
		return 0
	}
	pt.mu.Lock()
	defer pt.mu.Unlock()
	return pt.pausedUntilUnix
}

// ─── Plan Compression & Archival ────────────────────────────────────────────

// CompressedPlan represents a tightly packed, immutable plan history.
type CompressedPlan struct {
	Domain        string  `json:"d"`
	URLCount      int     `json:"c"`
	EncodedURLs   []byte  `json:"u"` // Gzip'd JSON array of URLs
	FinalPriority float64 `json:"p"`
	ArchivedAt    int64   `json:"a"`
	RunID         string  `json:"r"`
}

// CompressPlan encodes a finished plan for long-term archival to minimize storage.
func CompressPlan(plan *FrontierPlan) (*CompressedPlan, error) {
	if plan == nil {
		return nil, errors.New("plan is nil")
	}

	urlsJSON, err := json.Marshal(plan.URLQueue)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := gz.Write(urlsJSON); err != nil {
		return nil, err
	}
	if err := gz.Close(); err != nil {
		return nil, err
	}

	return &CompressedPlan{
		Domain:        plan.Domain,
		URLCount:      len(plan.URLQueue),
		EncodedURLs:   buf.Bytes(),
		FinalPriority: plan.Priority,
		ArchivedAt:    time.Now().Unix(),
		RunID:         plan.RunID,
	}, nil
}

// DecompressPlan unpacks an archived plan back into its active representation.
func DecompressPlan(compressed *CompressedPlan) (*FrontierPlan, error) {
	if compressed == nil {
		return nil, errors.New("compressed plan is nil")
	}

	gz, err := gzip.NewReader(bytes.NewReader(compressed.EncodedURLs))
	if err != nil {
		return nil, err
	}

	urlsJSON, err := io.ReadAll(gz)
	if err != nil {
		gz.Close()
		return nil, err
	}
	gz.Close()

	var urls []PlannedURL
	if err := json.Unmarshal(urlsJSON, &urls); err != nil {
		return nil, err
	}

	plan := &FrontierPlan{
		Domain:              compressed.Domain,
		URLQueue:            urls,
		Priority:            compressed.FinalPriority,
		PlanGeneratedAtUnix: compressed.ArchivedAt,
		RunID:               compressed.RunID,
	}
	plan.EstimatedSignal = estimatePlanSignal(urls)
	plan.ResumeToken = BuildResumeToken(plan)
	return plan, nil
}

// ─── Plan Dependency Tracking ───────────────────────────────────────────────

// PlanDependency represents a relationship where one plan must complete before
// another can be safely executed (e.g., fetching a login token before crawling
// an authenticated API).
type PlanDependency struct {
	SourceDomain string `json:"source_domain"`
	TargetDomain string `json:"target_domain"`
	RequiredRule string `json:"required_rule"`
	Satisfied    bool   `json:"satisfied"`
}

// DependencyGraph manages inter-plan dependencies to enforce execution order.
type DependencyGraph struct {
	mu           sync.RWMutex
	dependencies map[string][]PlanDependency
	completed    map[string]bool
}

// NewDependencyGraph creates an empty dependency tracker.
func NewDependencyGraph() *DependencyGraph {
	return &DependencyGraph{
		dependencies: make(map[string][]PlanDependency),
		completed:    make(map[string]bool),
	}
}

// AddDependency registers a new dependency rule.
func (g *DependencyGraph) AddDependency(target, source, rule string) {
	if g == nil || target == "" || source == "" {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	target = normalizeDomain(target)
	source = normalizeDomain(source)

	deps := g.dependencies[target]
	for _, d := range deps {
		if d.SourceDomain == source && d.RequiredRule == rule {
			return // Already exists
		}
	}

	g.dependencies[target] = append(deps, PlanDependency{
		SourceDomain: source,
		TargetDomain: target,
		RequiredRule: rule,
		Satisfied:    g.completed[source],
	})
}

// MarkCompleted flags a domain as successfully crawled, satisfying its dependents.
func (g *DependencyGraph) MarkCompleted(domain string) {
	if g == nil || domain == "" {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	domain = normalizeDomain(domain)
	g.completed[domain] = true

	// Update all targets that depend on this domain
	for target, deps := range g.dependencies {
		for i, d := range deps {
			if d.SourceDomain == domain {
				g.dependencies[target][i].Satisfied = true
			}
		}
	}
}

// CanExecute returns true if all dependencies for a target domain are satisfied.
func (g *DependencyGraph) CanExecute(target string) bool {
	if g == nil {
		return true // Fail open if no graph
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	target = normalizeDomain(target)

	deps, ok := g.dependencies[target]
	if !ok {
		return true // No dependencies
	}

	for _, d := range deps {
		if !d.Satisfied {
			return false
		}
	}
	return true
}

// GetPendingDependencies returns a list of unsatisfied source domains for a target.
func (g *DependencyGraph) GetPendingDependencies(target string) []string {
	if g == nil {
		return nil
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	target = normalizeDomain(target)

	deps, ok := g.dependencies[target]
	if !ok {
		return nil
	}

	var pending []string
	for _, d := range deps {
		if !d.Satisfied {
			pending = append(pending, d.SourceDomain)
		}
	}
	return pending
}

// StrictPlanValidationIssue captures a single planner invariant violation.
type StrictPlanValidationIssue struct {
	Field    string `json:"field"`
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
}

// StrictPlanValidationReport is used by interface/index_daemon before enqueuing a plan.
type StrictPlanValidationReport struct {
	Valid      bool                        `json:"valid"`
	IssueCount int                         `json:"issue_count"`
	Errors     int                         `json:"errors"`
	Warnings   int                         `json:"warnings"`
	Issues     []StrictPlanValidationIssue `json:"issues"`
}

// DetailedPlanDiff describes how a refreshed plan differs from the active one.
type DetailedPlanDiff struct {
	Domain           string   `json:"domain"`
	AddedURLs        []string `json:"added_urls"`
	RemovedURLs      []string `json:"removed_urls"`
	RetainedURLs     int      `json:"retained_urls"`
	PriorityDelta    float64  `json:"priority_delta"`
	SignalDelta      float64  `json:"signal_delta"`
	ConcurrencyDelta int      `json:"concurrency_delta"`
	RequiresRestart  bool     `json:"requires_restart"`
}

// ScheduleWindow is a deterministic execution window for frontier consumers.
type ScheduleWindow struct {
	URL       string `json:"url"`
	StartUnix int64  `json:"start_unix"`
	EndUnix   int64  `json:"end_unix"`
	Ordinal   int    `json:"ordinal"`
	FetchMode string `json:"fetch_mode"`
}

// ValidatePlan verifies the plan invariants that the Python bus bridge assumes.
func ValidatePlan(plan *FrontierPlan) StrictPlanValidationReport {
	report := StrictPlanValidationReport{Valid: true}
	add := func(field, code, severity, message string) {
		report.Issues = append(report.Issues, StrictPlanValidationIssue{Field: field, Code: code, Severity: severity, Message: message})
		report.IssueCount++
		if severity == "error" {
			report.Errors++
			report.Valid = false
		} else {
			report.Warnings++
		}
	}
	if plan == nil {
		add("plan", "nil_plan", "error", "plan must not be nil")
		return report
	}
	if normalizeDomain(plan.Domain) == "" {
		add("domain", "empty_domain", "error", "plan domain is required")
	}
	if plan.RunID == "" {
		add("run_id", "empty_run_id", "error", "run_id is required")
	}
	if plan.MaxConcurrency < MinFrontierConcurrency || plan.MaxConcurrency > MaxFrontierConcurrency {
		add("max_concurrency", "out_of_range", "error", "concurrency outside allowed range")
	}
	if plan.RateLimitMS <= 0 {
		add("rate_limit_ms", "non_positive", "error", "rate limit must be positive")
	}
	if len(plan.URLQueue) == 0 {
		add("url_queue", "empty_queue", "warning", "plan has no URLs")
	}
	seen := make(map[string]bool, len(plan.URLQueue))
	for i, u := range plan.URLQueue {
		if u.URL == "" || !strings.HasPrefix(u.URL, "http") {
			add("url_queue", "invalid_url", "error", "planned URL must be absolute http(s)")
		}
		if seen[u.URL] {
			add("url_queue", "duplicate_url", "error", "planned URL appears more than once")
		}
		seen[u.URL] = true
		if u.ResumeOrdinal != i {
			add("resume_ordinal", "ordinal_mismatch", "warning", "resume ordinal does not match queue order")
		}
		if u.SignalExpectation < 0 || u.SignalExpectation > 1 {
			add("signal_expectation", "out_of_range", "error", "signal expectation must be in 0..1")
		}
	}
	return report
}

// DiffPlansDetailed computes URL and scheduling differences between plan generations.
func DiffPlansDetailed(oldPlan, newPlan *FrontierPlan) DetailedPlanDiff {
	if oldPlan == nil && newPlan == nil {
		return DetailedPlanDiff{}
	}
	if oldPlan == nil {
		return DetailedPlanDiff{Domain: newPlan.Domain, AddedURLs: plannedURLStrings(newPlan.URLQueue), PriorityDelta: newPlan.Priority, SignalDelta: newPlan.EstimatedSignal, RequiresRestart: true}
	}
	if newPlan == nil {
		return DetailedPlanDiff{Domain: oldPlan.Domain, RemovedURLs: plannedURLStrings(oldPlan.URLQueue), PriorityDelta: -oldPlan.Priority, SignalDelta: -oldPlan.EstimatedSignal, RequiresRestart: true}
	}
	oldURLs := plannedURLSet(oldPlan.URLQueue)
	newURLs := plannedURLSet(newPlan.URLQueue)
	diff := DetailedPlanDiff{
		Domain:           firstNonEmptyString(newPlan.Domain, oldPlan.Domain),
		PriorityDelta:    newPlan.Priority - oldPlan.Priority,
		SignalDelta:      newPlan.EstimatedSignal - oldPlan.EstimatedSignal,
		ConcurrencyDelta: newPlan.MaxConcurrency - oldPlan.MaxConcurrency,
	}
	for u := range newURLs {
		if !oldURLs[u] {
			diff.AddedURLs = append(diff.AddedURLs, u)
		} else {
			diff.RetainedURLs++
		}
	}
	for u := range oldURLs {
		if !newURLs[u] {
			diff.RemovedURLs = append(diff.RemovedURLs, u)
		}
	}
	sort.Strings(diff.AddedURLs)
	sort.Strings(diff.RemovedURLs)
	diff.RequiresRestart = len(diff.RemovedURLs) > 0 || diff.ConcurrencyDelta < 0 || math.Abs(diff.PriorityDelta) > 0.25
	return diff
}

// BuildScheduleWindows deterministically assigns URL execution windows.
func BuildScheduleWindows(plan *FrontierPlan, startUnix int64) []ScheduleWindow {
	if plan == nil {
		return nil
	}
	if startUnix <= 0 {
		startUnix = time.Now().Unix()
	}
	delay := plan.RateLimitMS
	if delay <= 0 {
		delay = 500
	}
	concurrency := plan.MaxConcurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	windows := make([]ScheduleWindow, 0, len(plan.URLQueue))
	for i, u := range plan.URLQueue {
		slot := i / concurrency
		start := startUnix + int64(slot)*(delay/1000)
		if delay < 1000 {
			start = startUnix + int64(slot)
		}
		windows = append(windows, ScheduleWindow{
			URL:       u.URL,
			StartUnix: start,
			EndUnix:   start + maxPlanInt64(1, delay/1000),
			Ordinal:   i,
			FetchMode: u.FetchMode,
		})
	}
	return windows
}

func plannedURLSet(urls []PlannedURL) map[string]bool {
	out := make(map[string]bool, len(urls))
	for _, u := range urls {
		if u.URL != "" {
			out[u.URL] = true
		}
	}
	return out
}

func plannedURLStrings(urls []PlannedURL) []string {
	out := make([]string, 0, len(urls))
	for _, u := range urls {
		if u.URL != "" {
			out = append(out, u.URL)
		}
	}
	sort.Strings(out)
	return out
}

func maxPlanInt64(a int64, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
