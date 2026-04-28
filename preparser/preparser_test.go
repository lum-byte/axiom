package preparser

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAnalyzeDomainBuildsFingerprintAndEvent(t *testing.T) {
	a := NewDomainAnalyzer()
	m, err := a.AnalyzeDomain("https://Example.com", []string{"/docs/api", "/blog/post"}, "User-agent: *\nCrawl-delay: 2\nDisallow: /admin", []string{"https://example.com/sitemap.xml"}, "123e4567-e89b-12d3-a456-426614174000")
	if err != nil {
		t.Fatal(err)
	}
	if m.Domain != "example.com" {
		t.Fatalf("domain normalization failed: %q", m.Domain)
	}
	if m.RateLimitProfile.RequestsPerSecond != 0.5 {
		t.Fatalf("unexpected rps: %v", m.RateLimitProfile.RequestsPerSecond)
	}
	if _, ok := a.ReadFingerprint("example.com"); !ok {
		t.Fatal("fingerprint missing")
	}
	req := m.BridgeEvent()
	if req.Topic != "domain_topology" {
		t.Fatalf("wrong topic: %s", req.Topic)
	}
	if _, err := EncodeBridgeRequest(req); err != nil {
		t.Fatal(err)
	}
}

func TestPlanCrawlRanksAndSetsFetchMode(t *testing.T) {
	dm := DomainMap{Domain: "example.com", PathTopologyMap: map[string]string{"/app/*": "SAAS_DOCS"}, RenderRequirements: map[string]string{"/app/*": "headless"}, RateLimitProfile: RateLimitProfile{Domain: "example.com", RequestsPerSecond: 2, BurstCapacity: 4}}
	plan, err := PlanCrawl(CrawlPlanInput{Domain: "example.com", CandidateURLs: []string{"/app/guide", "https://other.test/no"}, DomainMap: dm, PhaseWeight: 2, FreshnessDecay: 1, FrictionForecast: 0.8, RunID: "123e4567-e89b-12d3-a456-426614174000"})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Manifest.TotalURLs != 1 {
		t.Fatalf("expected one domain URL, got %d", plan.Manifest.TotalURLs)
	}
	if plan.Manifest.ClearanceRequired != 3 {
		t.Fatalf("expected tor clearance, got %d", plan.Manifest.ClearanceRequired)
	}
	if plan.BridgeEvent().Topic != "crawl_manifest" {
		t.Fatal("wrong bridge topic")
	}
}

func TestGeneratePlanSerializesAndPrioritizes(t *testing.T) {
	fp := &DomainFingerprint{
		Domain:               "example.com",
		TopologyClass:        TopologySaaSDocs,
		TopologyDistribution: map[string]float64{TopologySaaSDocs: 0.8, TopologyGenericHTML: 0.2},
		URLPatterns: []URLPattern{
			{Pattern: "/docs/*", Count: 10, TopologyClass: TopologySaaSDocs, Confidence: 0.9, Examples: []string{"/docs/intro"}},
			{Pattern: "/api/*", Count: 4, TopologyClass: TopologyRESTAPIJSON, Confidence: 0.6},
		},
		RobotsSignals:       RobotsAnalysis{CrawlDelaySeconds: 1, FrictionLevel: FrictionLevelCL2, SitemapURLs: []string{"https://example.com/sitemap.xml"}},
		PhaseRecommendation: PhaseRecommendationLearning,
		FrictionLevel:       FrictionLevelCL2,
		SignalDensity:       0.7,
		ObservationCount:    20,
		RunID:               "123e4567-e89b-12d3-a456-426614174000",
	}
	plan, err := GeneratePlan(fp, PlanOptions{MaxURLs: 4, IncludeSitemaps: true, QueryHints: []string{"install"}, DaysSinceLastCrawl: 1})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Domain != "example.com" || len(plan.URLQueue) == 0 {
		t.Fatalf("unexpected plan: %#v", plan)
	}
	if plan.MaxConcurrency != MaxConcurrencyForFriction(FrictionLevelCL2) {
		t.Fatalf("unexpected concurrency: %d", plan.MaxConcurrency)
	}
	if _, err := DecodeResumeToken(plan.ResumeToken); err != nil {
		t.Fatal(err)
	}
	wire, err := SerializePlan(plan)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := DeserializePlan(wire)
	if err != nil {
		t.Fatal(err)
	}
	if roundTrip.Domain != plan.Domain {
		t.Fatal("plan roundtrip lost domain")
	}
	plans := PrioritizePlan([]*FrontierPlan{roundTrip, &FrontierPlan{Domain: "low", Priority: 0.01}})
	if plans[0].Domain != "example.com" {
		t.Fatal("priority order failed")
	}
	if plan.BridgeEvent().Topic != "crawl_manifest" {
		t.Fatal("wrong bridge topic")
	}
}

func TestPlannerFormulaHelpers(t *testing.T) {
	if PhaseWeight(PhaseRecommendationKnown) != 1.0 {
		t.Fatal("known phase should have full weight")
	}
	if FreshnessDecay(0, 0) != 1.0 {
		t.Fatal("freshness with zero days should be one")
	}
	if RateLimitDelayMS(0, FrictionLevelCL4) < 5000 {
		t.Fatal("CL4 needs conservative delay")
	}
	if plannedFetchMode("headless", FrictionLevelCL1, false) != "headless" {
		t.Fatal("headless render should request headless fetch")
	}
}

func TestExtractSignalClassifiesCode(t *testing.T) {
	event, err := ExtractSignal(SignalExtractionInput{URL: "https://example.com", TopologyClass: "SAAS_DOCS", SanitizedText: "func main() {\n return nil\n}", RunID: "123e4567-e89b-12d3-a456-426614174000"})
	if err != nil {
		t.Fatal(err)
	}
	if event.SignalType != "code" {
		t.Fatalf("expected code, got %s", event.SignalType)
	}
	if event.TokenCount == 0 || event.SignalDensity <= 0 {
		t.Fatal("expected non-zero signal stats")
	}
}

func TestExtractStructuredSignalSplitsZones(t *testing.T) {
	text := "# Install\n\nUse this package to parse docs.\n\n```go\nfunc main() {\n return\n}\n```\n\n| name | value |\n| --- | --- |\n| a | b |"
	extracted, err := ExtractStructuredSignal(SignalExtractionInput{
		URL:           "https://example.com/docs/install",
		TopologyClass: TopologySaaSDocs,
		SanitizedText: text,
		RunID:         "123e4567-e89b-12d3-a456-426614174000",
	}, ZoneExtractionOptions{PreferCodeLanguage: true})
	if err != nil {
		t.Fatal(err)
	}
	if extracted.Domain != "example.com" || len(extracted.Zones) < 3 {
		t.Fatalf("unexpected extracted signal: %#v", extracted)
	}
	foundCode := false
	foundTable := false
	for _, zone := range extracted.Zones {
		if zone.Type == ZoneCode {
			foundCode = true
		}
		if zone.Type == ZoneTable {
			foundTable = true
		}
	}
	if !foundCode || !foundTable {
		t.Fatalf("expected code and table zones: %#v", extracted.Zones)
	}
	if extracted.Event().SignalType == "" || extracted.BridgeEvent().Topic != "signal_extracted" {
		t.Fatal("event conversion failed")
	}
}

func TestSignalZoneFilteringAndRanking(t *testing.T) {
	text := "Home Login Privacy Terms Menu\n\nImportant result paragraph with enough words to survive filtering.\n\n- first\n- second\n- third"
	extracted, err := ExtractStructuredSignal(SignalExtractionInput{
		URL:           "https://example.com/article",
		TopologyClass: TopologyNewsArticle,
		SanitizedText: text,
		RunID:         "123e4567-e89b-12d3-a456-426614174000",
	}, ZoneExtractionOptions{AllowedTypes: []ZoneType{ZoneProse, ZoneList}})
	if err != nil {
		t.Fatal(err)
	}
	for _, zone := range extracted.Zones {
		if zone.Type == ZoneNavigation {
			t.Fatal("navigation should have been filtered")
		}
		if zone.Rank == 0 {
			t.Fatal("zone rank not assigned")
		}
	}
	if extracted.SignalDensity <= 0 {
		t.Fatal("expected signal density")
	}
}

func TestValidateRecipeStale(t *testing.T) {
	health, stale, err := ValidateRecipe("NEWS_ARTICLE", "grep article", []RecipeValidationSample{{CleanSignal: "", LatencyMS: 10, Succeeded: false}, {CleanSignal: "ok", LatencyMS: 20, Succeeded: true}}, "123e4567-e89b-12d3-a456-426614174000")
	if err != nil {
		t.Fatal(err)
	}
	if !health.Stale || stale == nil {
		t.Fatal("expected stale recipe")
	}
	body, err := json.Marshal(stale.BridgeEvent())
	if err != nil || len(body) == 0 {
		t.Fatalf("bridge marshal failed: %v", err)
	}
}

func TestValidateRecipeWindowDetectsYieldDrop(t *testing.T) {
	record := RecipeRegistryRecord{Domain: "example.com", TopologyClass: TopologyNewsArticle, Recipe: "article main", HistoricalYield: 0.10, SampleCount: 50}
	samples := []RecipeYieldSample{
		{URL: "https://example.com/a", RawBytes: 10000, SignalBytes: 100, LatencyMS: 15, Succeeded: true, CapturedAtUnix: 1},
		{URL: "https://example.com/b", RawBytes: 10000, SignalBytes: 90, LatencyMS: 12, Succeeded: true, CapturedAtUnix: 2},
		{URL: "https://example.com/c", RawBytes: 10000, SignalBytes: 0, LatencyMS: 11, Succeeded: false, CapturedAtUnix: 3},
	}
	report, health, stale, err := ValidateRecipeWindow(record, samples, RecipeValidationOptions{RunID: "123e4567-e89b-12d3-a456-426614174000"})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Stale || stale == nil || !health.Stale {
		t.Fatalf("expected stale report: %#v", report)
	}
	if report.RecommendedAction == "" || report.RecommendedPriority > 1 {
		t.Fatal("expected recompile recommendation")
	}
	updated := ApplyValidationToRegistry(record, report)
	if !updated.Stale || updated.SampleCount != 53 {
		t.Fatalf("unexpected registry update: %#v", updated)
	}
}

func TestEvaluateRecipeAgainstSignals(t *testing.T) {
	record := RecipeRegistryRecord{Domain: "example.com", TopologyClass: TopologySaaSDocs, Recipe: "main docs", HistoricalYield: 0.01}
	raw := map[string]string{
		"https://example.com/docs/a": strings.Repeat("raw ", 100),
		"https://example.com/docs/b": strings.Repeat("raw ", 200),
	}
	signals := map[string]string{
		"https://example.com/docs/a": "signal content",
		"https://example.com/docs/b": "signal content with more words",
	}
	report, err := EvaluateRecipeAgainstSignals(record, raw, signals, RecipeValidationOptions{RunID: "123e4567-e89b-12d3-a456-426614174000"})
	if err != nil {
		t.Fatal(err)
	}
	if report.SampleSize != 2 || report.MeanYield <= 0 {
		t.Fatalf("unexpected report: %#v", report)
	}
}

func TestAnalyzeFetchRecordsProducesDomainFingerprint(t *testing.T) {
	a := NewDomainAnalyzer()
	records := []FetchRecord{
		{URL: "https://example.com/docs/intro", StatusCode: 200, ContentType: "text/html", ContentLanguage: "en", ResponseBytes: 4096, LatencyMS: 25, FetchedAtUnix: 1700000000},
		{URL: "https://example.com/docs/api", StatusCode: 200, ContentType: "text/html", ContentLanguage: "en", ResponseBytes: 8192, LatencyMS: 35, FetchedAtUnix: 1700000100},
		{URL: "https://example.com/api/v1/items", StatusCode: 200, ContentType: "application/json", ResponseBytes: 2048, LatencyMS: 15, FetchedAtUnix: 1700000200},
	}
	fp, err := a.AnalyzeFetchRecords("example.com", records, "User-agent: *\nCrawl-delay: 1\nSitemap: https://example.com/sitemap.xml", nil, "123e4567-e89b-12d3-a456-426614174000")
	if err != nil {
		t.Fatal(err)
	}
	if fp.Domain != "example.com" {
		t.Fatalf("wrong domain: %s", fp.Domain)
	}
	if fp.ObservationCount != len(records) {
		t.Fatalf("wrong observation count: %d", fp.ObservationCount)
	}
	if fp.RobotsSignals.FrictionLevel < FrictionLevelCL1 {
		t.Fatal("expected robots friction")
	}
	if len(fp.URLPatterns) == 0 {
		t.Fatal("expected URL patterns")
	}
	if ValidateDomainFingerprint(fp).Valid != true {
		t.Fatal("fingerprint should validate")
	}
	if fp.BridgeEvent().Topic != "domain_topology" {
		t.Fatal("wrong bridge topic")
	}
}

func TestFetchRecordValidationAndJSONL(t *testing.T) {
	body := `{"url":"https://example.com/a","status_code":200,"content_type":"text/html","response_bytes":10,"latency_ms":2}` + "\n" +
		`{"url":"https://example.com/b","status_code":404,"response_bytes":0,"latency_ms":3}` + "\n"
	records, err := ParseFetchRecordJSONL(strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	report := ValidateFetchRecords(records)
	if !report.Valid {
		t.Fatalf("records should validate: %#v", report.Issues)
	}
	if report.Accepted != 2 || report.DomainCount["example.com"] != 2 {
		t.Fatalf("unexpected report: %#v", report)
	}
	bad := ValidateFetchRecords([]FetchRecord{{URL: "ftp://example.com/a", StatusCode: 200, ResponseBytes: -1, LatencyMS: -1}})
	if bad.Valid || bad.ErrorCount == 0 {
		t.Fatal("expected validation errors")
	}
}

func TestPatternTrieCompressesVariableSegments(t *testing.T) {
	patterns := BuildURLPatterns([]string{"/product/100", "/product/101", "/product/102", "/docs/intro"}, map[string]int{"ECOMMERCE_PRODUCT": 3})
	if len(patterns) == 0 {
		t.Fatal("expected patterns")
	}
	foundVariable := false
	for _, pattern := range patterns {
		if pattern.Pattern == "/product/*/*" || pattern.Pattern == "/product/*" {
			foundVariable = true
		}
	}
	if !foundVariable {
		t.Fatalf("expected variable product pattern, got %#v", patterns)
	}
}

func TestFingerprintDriftAndLearningHints(t *testing.T) {
	prev := &DomainFingerprint{
		Domain:               "example.com",
		TopologyClass:        TopologyGenericHTML,
		TopologyDistribution: map[string]float64{TopologyGenericHTML: 1},
		URLPatterns:          []URLPattern{{Pattern: "/docs/*", Count: 10, TopologyClass: TopologySaaSDocs, Confidence: 0.8}},
		FrictionLevel:        FrictionLevelCL1,
		Confidence:           0.8,
		SignalDensity:        0.7,
		FingerprintSHA256:    "prev",
	}
	curr := &DomainFingerprint{
		Domain:               "example.com",
		TopologyClass:        TopologyRESTAPIJSON,
		TopologyDistribution: map[string]float64{TopologyRESTAPIJSON: 0.8, TopologyGenericHTML: 0.2},
		URLPatterns:          []URLPattern{{Pattern: "/api/*", Count: 25, TopologyClass: TopologyRESTAPIJSON, Confidence: 0.9}},
		FrictionLevel:        FrictionLevelCL3,
		Confidence:           0.6,
		SignalDensity:        0.3,
		FingerprintSHA256:    "curr",
		ObservationCount:     25,
	}
	drift := CompareDomainFingerprints(prev, curr)
	if drift.DriftScore <= 0 || !drift.PlanRefreshNeeded {
		t.Fatalf("expected drift plan refresh: %#v", drift)
	}
	hints := BuildLearningHints(curr)
	if len(hints) != 0 {
		t.Fatalf("REST_API_JSON with enough observations should not require generic hints: %#v", hints)
	}
}

func TestMemoryCursorStoreBatchAnalyze(t *testing.T) {
	store := NewMemoryCursorStore()
	store.PutDomainHistory("example.com", []FetchRecord{
		{URL: "https://example.com/news/a", StatusCode: 200, ContentType: "text/html", ResponseBytes: 2048, LatencyMS: 11, FetchedAtUnix: 1700000000},
		{URL: "https://example.com/news/b", StatusCode: 200, ContentType: "text/html", ResponseBytes: 4096, LatencyMS: 12, FetchedAtUnix: 1700000001},
	})
	store.PutRobots("example.com", "User-agent: *\nAllow: /")
	store.PutSitemaps("example.com", []string{"https://example.com/sitemap.xml"})
	fps, err := NewDomainAnalyzer().BatchAnalyzeStore([]string{"example.com"}, store, "123e4567-e89b-12d3-a456-426614174000")
	if err != nil {
		t.Fatal(err)
	}
	if len(fps) != 1 || fps[0].Domain != "example.com" {
		t.Fatalf("unexpected fingerprints: %#v", fps)
	}
	if !strings.Contains(FingerprintSummaryLine(fps[0]), "domain=example.com") {
		t.Fatal("summary line missing domain")
	}
}
