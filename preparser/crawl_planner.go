package preparser

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"net/url"
	"sort"
	"strconv"
	"strings"
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
	MaxURLs              int               `json:"max_urls"`
	Phase                string            `json:"phase"`
	DaysSinceLastCrawl   float64           `json:"days_since_last_crawl"`
	FreshnessLambda      float64           `json:"freshness_lambda"`
	NowUnix             int64             `json:"now_unix"`
	MaxConcurrency       int               `json:"max_concurrency"`
	IncludeRobots        bool              `json:"include_robots"`
	IncludeSitemaps      bool              `json:"include_sitemaps"`
	QueryHints           []string          `json:"query_hints"`
	SeenURLHashes        map[string]bool   `json:"seen_url_hashes"`
	TopologyWeights      map[string]float64 `json:"topology_weights"`
	AllowHighFriction    bool              `json:"allow_high_friction"`
	ResumeAfterURL       string            `json:"resume_after_url"`
	ExpectedSignalFloor  float64           `json:"expected_signal_floor"`
	PreferFreshPatterns  bool              `json:"prefer_fresh_patterns"`
	FrontierComponent    string            `json:"frontier_component"`
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
