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

func TestPlanValidationDiffAndScheduleWindows(t *testing.T) {
	plan := &FrontierPlan{
		Domain:         "example.com",
		Priority:       0.8,
		RateLimitMS:    1000,
		MaxConcurrency: 2,
		FrictionLevel:  FrictionLevelCL2,
		RunID:          "123e4567-e89b-12d3-a456-426614174000",
		URLQueue: []PlannedURL{
			{URL: "https://example.com/a", SignalExpectation: 0.8, ResumeOrdinal: 0, FetchMode: "static"},
			{URL: "https://example.com/b", SignalExpectation: 0.7, ResumeOrdinal: 1, FetchMode: "headless"},
		},
		EstimatedSignal: 0.75,
	}
	report := ValidatePlan(plan)
	if !report.Valid {
		t.Fatalf("plan should validate: %#v", report)
	}
	next := *plan
	next.URLQueue = append([]PlannedURL(nil), plan.URLQueue...)
	next.Priority = 0.3
	next.URLQueue = append(next.URLQueue[:1], PlannedURL{URL: "https://example.com/c", SignalExpectation: 0.9, ResumeOrdinal: 1, FetchMode: "static"})
	diff := DiffPlansDetailed(plan, &next)
	if len(diff.AddedURLs) != 1 || len(diff.RemovedURLs) != 1 || !diff.RequiresRestart {
		t.Fatalf("unexpected diff: %#v", diff)
	}
	windows := BuildScheduleWindows(plan, 100)
	if len(windows) != 2 || windows[0].StartUnix != windows[1].StartUnix {
		t.Fatalf("unexpected windows: %#v", windows)
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

func TestRecipeDSLCompileExecuteTrendAndSuggestions(t *testing.T) {
	recipe := "keep_between:<main> => </main>; drop:nav; strip_html; collapse_ws; min_tokens:2"
	compiled, lint := CompileRecipe(TopologyNewsArticle, recipe, 10)
	if !lint.Valid {
		t.Fatalf("recipe should lint: %#v", lint)
	}
	exec := ExecuteCompiledRecipe(compiled, "<html><nav>bad</nav><main><h1>Hello</h1> signal body</main></html>", "https://example.com/a")
	if exec.Rejected || !strings.Contains(exec.Signal, "Hello signal body") {
		t.Fatalf("unexpected execution: %#v", exec)
	}
	record := RecipeRegistryRecord{Domain: "example.com", TopologyClass: TopologyNewsArticle, Recipe: recipe, HistoricalYield: 0.05}
	report, results, err := EvaluateCompiledRecipe(record, map[string]string{"https://example.com/a": "<main>good signal</main>"}, RecipeValidationOptions{RunID: "123e4567-e89b-12d3-a456-426614174000"})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || report.SampleSize != 1 {
		t.Fatalf("unexpected compiled evaluation: %#v %#v", report, results)
	}
	trend := BuildRecipeTrend(record, []RecipeYieldSample{
		{URL: "a", RawBytes: 100, SignalBytes: 20, Succeeded: true, CapturedAtUnix: 100},
		{URL: "b", RawBytes: 100, SignalBytes: 10, Succeeded: true, CapturedAtUnix: 200},
		{URL: "c", RawBytes: 100, SignalBytes: 2, Succeeded: true, CapturedAtUnix: 90000},
	}, 86400)
	if len(trend.Points) == 0 {
		t.Fatal("expected trend points")
	}
	suggestions := SuggestRecipePatches(RecipeValidationReport{TopologyClass: TopologyNewsArticle, RecipeHash: compiled.Hash, EmptyRate: 0.5, FailureRate: 0.1, MeanYield: 0.001, MedianLatencyMS: 10}, lint)
	if len(suggestions) == 0 {
		t.Fatal("expected patch suggestions")
	}
}

func TestRecipeRegistryBatchSnapshotAndLintFailures(t *testing.T) {
	_, lint := CompileRecipe(TopologySaaSDocs, "keep_between:only-left", 0)
	if lint.Valid || lint.ErrorCount == 0 {
		t.Fatal("expected lint error")
	}
	record := RecipeRegistryRecord{Domain: "example.com", TopologyClass: TopologySaaSDocs, Recipe: "strip_html; collapse_ws", HistoricalYield: 0.02, HistoricalLatencyMS: 12}
	snapshot := ValidateRecipeRegistry([]RecipeRegistryRecord{record})
	if snapshot.RecordCount != 1 || snapshot.ByTopology[TopologySaaSDocs] != 1 || snapshot.SnapshotHash == "" {
		t.Fatalf("unexpected snapshot: %#v", snapshot)
	}
	key := recipeRegistryKey(record)
	batch := ValidateRegistryBatch([]RecipeRegistryRecord{record}, map[string][]RecipeYieldSample{
		key: []RecipeYieldSample{{URL: "https://example.com", RawBytes: 1000, SignalBytes: 20, Succeeded: true, CapturedAtUnix: 1}},
	}, RecipeValidationOptions{RunID: "123e4567-e89b-12d3-a456-426614174000"})
	if len(batch.Reports) != 1 || len(batch.RegistryUpdates) != 1 {
		t.Fatalf("unexpected batch: %#v", batch)
	}
}

func TestRecipeAdvancedValidationUtilities(t *testing.T) {
	zones := []SignalZone{
		{Type: ZoneCode, Content: "func main() {}", TokenCount: 4, Density: 0.8, TopologyClass: TopologySaaSDocs},
		{Type: ZoneProse, Content: "documentation content", TokenCount: 10, Density: 0.7, TopologyClass: TopologySaaSDocs},
	}
	candidates := MineSelectorCandidates(TopologySaaSDocs, zones)
	if len(candidates) == 0 {
		t.Fatal("expected selector candidates")
	}
	draft := DraftRecipeFromZones(TopologySaaSDocs, zones, 3)
	if !strings.Contains(draft, "select:") {
		t.Fatalf("unexpected draft: %s", draft)
	}
	report := RecipeValidationReport{MeanYield: 0.5, FailureRate: 0.1, EmptyRate: 0.1, TooBroadRate: 0.4, MedianLatencyMS: 50, SampleSize: 4}
	mutations := MutateRecipe("select:main", report)
	if len(mutations) == 0 {
		t.Fatal("expected recipe mutations")
	}
	compiled, lint := ApplyRecipeMutation(mutations[0])
	if compiled.Raw == "" || !lint.Valid {
		t.Fatalf("mutation should compile: %#v %#v", compiled, lint)
	}
	raw := map[string]string{"u1": "<main>good signal</main>", "u2": "<main>other signal</main>"}
	ab := CompareRecipesAB(RecipeRegistryRecord{Domain: "example.com", TopologyClass: TopologySaaSDocs}, "strip_html; collapse_ws", draft, raw, RecipeValidationOptions{RunID: "123e4567-e89b-12d3-a456-426614174000"})
	if ab.Winner == "" {
		t.Fatalf("expected winner: %#v", ab)
	}
	results := []RecipeExecutionResult{{URL: "u1", RawBytes: 100, SignalBytes: 20}, {URL: "u2", RawBytes: 100, SignalBytes: 0, Rejected: true}}
	coverage := ComputeRecipeCoverage(results, "example.com", TopologySaaSDocs, compiled.Hash)
	if coverage.URLsSeen != 2 || coverage.CoverageRatio <= 0 {
		t.Fatalf("unexpected coverage: %#v", coverage)
	}
	split := SplitRecipeSamples([]RecipeYieldSample{{URL: "1"}, {URL: "2"}, {URL: "3"}, {URL: "4"}, {URL: "5"}}, 0.2, 0.2)
	if len(split.Train) == 0 || len(split.Validation) == 0 || len(split.Test) == 0 {
		t.Fatalf("unexpected split: %#v", split)
	}
	cv := CrossValidateRecipe(RecipeRegistryRecord{Domain: "example.com", TopologyClass: TopologySaaSDocs, Recipe: "strip_html"}, []RecipeYieldSample{{RawBytes: 100, SignalBytes: 10, Succeeded: true}, {RawBytes: 100, SignalBytes: 12, Succeeded: true}, {RawBytes: 100, SignalBytes: 11, Succeeded: true}}, 3, RecipeValidationOptions{})
	if len(cv.FoldReports) != 3 {
		t.Fatalf("unexpected cross validation: %#v", cv)
	}
	oldRecords := []RecipeRegistryRecord{{Domain: "example.com", TopologyClass: TopologySaaSDocs, RecipeHash: "old", Recipe: "a"}}
	newRecords := []RecipeRegistryRecord{{Domain: "example.com", TopologyClass: TopologySaaSDocs, RecipeHash: "new", Recipe: "b"}}
	diff := DiffRecipeRegistries(oldRecords, newRecords)
	if !diff.RequiresReload || len(diff.Changed) != 1 {
		t.Fatalf("unexpected registry diff: %#v", diff)
	}
	wire, err := SerializeRecipeRegistry(newRecords)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := DeserializeRecipeRegistry(wire)
	if err != nil || len(roundTrip) != 1 {
		t.Fatalf("registry roundtrip failed: %v %#v", err, roundTrip)
	}
	gate := GateRecipePromotion(RecipeValidationReport{MeanYield: 0.05, FailureRate: 0, EmptyRate: 0, SampleSize: 2}, RecipeLintReport{Valid: true}, coverage, 0.1)
	if !gate.Promote {
		t.Fatalf("expected promotion: %#v", gate)
	}
	risk := AssessRecipeRisk(CompiledRecipe{Hash: "h", Steps: []RecipeStep{{Kind: RecipeStepRegexCapture, Argument: "(.*foo.*bar.*)"}}}, RecipeLintReport{Valid: false})
	if !risk.RequiresReview {
		t.Fatalf("expected risk: %#v", risk)
	}
}

func TestRecipeRegistryMaintenanceRolloutAndRepair(t *testing.T) {
	compiled, lint := CompileRecipe(TopologySaaSDocs, "strip_html; collapse_ws; min_tokens:1", 1)
	raw := map[string]string{"u1": "<main>good signal</main>", "u2": "<main>more signal</main>", "u3": "<nav>bad</nav>"}
	bench := BenchmarkRecipe(compiled, raw)
	if bench.SampleCount != 3 || bench.MeanYield <= 0 {
		t.Fatalf("unexpected benchmark: %#v", bench)
	}
	results := []RecipeExecutionResult{
		{URL: "u1", RawBytes: 100, SignalBytes: 20},
		{URL: "u2", RawBytes: 100, SignalBytes: 0, RejectReason: "empty_signal"},
	}
	clusters := ClusterRecipeFailures(results)
	if len(clusters) != 1 || clusters[0].RecommendedStep == "" {
		t.Fatalf("unexpected clusters: %#v", clusters)
	}
	report := RecipeValidationReport{RecipeHash: compiled.Hash, MeanYield: 0.03, FailureRate: 0.1, EmptyRate: 0.2, MedianLatencyMS: 20, SampleSize: 2, Confidence: 0.8}
	coverage := ComputeRecipeCoverage(results, "example.com", TopologySaaSDocs, compiled.Hash)
	trend := BuildRecipeTrend(RecipeRegistryRecord{Domain: "example.com", TopologyClass: TopologySaaSDocs, Recipe: compiled.Raw}, []RecipeYieldSample{{RawBytes: 100, SignalBytes: 10, Succeeded: true, CapturedAtUnix: 1}, {RawBytes: 100, SignalBytes: 12, Succeeded: true, CapturedAtUnix: 2}}, 10)
	risk := AssessRecipeRisk(compiled, lint)
	score := ScoreRecipeQuality(report, coverage, trend, risk)
	if score.OverallScore <= 0 || score.Grade == "" {
		t.Fatalf("unexpected score: %#v", score)
	}
	gate := GateRecipePromotion(report, lint, coverage, 0.1)
	rollout := BuildRecipeRolloutPlan(gate, risk, bench)
	if rollout.Decision == "" {
		t.Fatalf("unexpected rollout: %#v", rollout)
	}
	repair := BuildRecipeRepairPlan(RecipeValidationReport{RecipeHash: compiled.Hash, EmptyRate: 0.5, SampleSize: 2, Confidence: 0.7}, lint, results, compiled.Raw)
	if len(repair.Actions) == 0 || repair.Confidence <= 0 {
		t.Fatalf("unexpected repair: %#v", repair)
	}
	records := []RecipeRegistryRecord{
		{Domain: "example.com", TopologyClass: TopologySaaSDocs, Recipe: compiled.Raw, HistoricalYield: 0.01, LastValidatedUnix: 1, Stale: true, SampleCount: 1},
		{Domain: "news.example.com", TopologyClass: TopologyNewsArticle, Recipe: "strip_html", HistoricalYield: 0.1, LastValidatedUnix: 100, SampleCount: 30},
	}
	ranked := RankRegistryRecords(records)
	if !ranked[0].Stale {
		t.Fatalf("expected stale first: %#v", ranked)
	}
	candidates := SelectRecipesForRevalidation(records, 1, 1000000)
	if len(candidates) != 1 || candidates[0].Priority <= 0 {
		t.Fatalf("unexpected candidates: %#v", candidates)
	}
	pruned := PruneStaleRecipes(records, 10, 1000000)
	if len(pruned) >= len(records) {
		t.Fatalf("expected prune: %#v", pruned)
	}
	events := GenerateRecipeHealthEvents(records, "123e4567-e89b-12d3-a456-426614174000")
	if len(events) != 2 {
		t.Fatalf("unexpected health events: %#v", events)
	}
	manifest := BuildRecipeManifest(records, 1)
	if len(manifest.Entries) != 2 || manifest.ManifestHash == "" {
		t.Fatalf("unexpected manifest: %#v", manifest)
	}
	merged := MergeRecipeReports([]RecipeValidationReport{report, report})
	if merged.SampleSize != report.SampleSize*2 {
		t.Fatalf("unexpected merged report: %#v", merged)
	}
	compat := CheckRecipeCompatibility(records[1], map[string]bool{TopologyNewsArticle: true})
	if !compat.Compatible {
		t.Fatalf("expected compatibility: %#v", compat)
	}
	migration := BuildRecipeMigrationPlan(records[:1], records)
	if len(migration.Steps) == 0 || migration.PlanHash == "" {
		t.Fatalf("unexpected migration: %#v", migration)
	}
	oldManifest := BuildRecipeManifest(records[:1], 1)
	diffManifest := DiffRecipeManifests(oldManifest, manifest)
	if !diffManifest.RequiresRestart || len(diffManifest.Added) != 1 {
		t.Fatalf("unexpected manifest diff: %#v", diffManifest)
	}
	rolled := ApplyRolloutDecision(records[0], RecipeRolloutPlan{Decision: "promote"}, 123)
	if rolled.Stale || rolled.LastValidatedUnix != 123 {
		t.Fatalf("unexpected rollout apply: %#v", rolled)
	}
	synth := SynthesizeValidationSamples(results, 42)
	if len(synth) != len(results) || synth[0].CapturedAtUnix != 42 {
		t.Fatalf("unexpected synthesized samples: %#v", synth)
	}
	audit := BuildRecipeAuditTrail(records, "validate", "test", "unit", 99)
	if len(audit.Entries) != len(records) || audit.Digest == "" {
		t.Fatalf("unexpected audit: %#v", audit)
	}
	invariants := CheckRecipeInvariants(records)
	if !invariants.Healthy {
		t.Fatalf("expected healthy invariants: %#v", invariants)
	}
	if !RecipeRecordsEqual(records[0], records[0]) {
		t.Fatal("record equality failed")
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

func TestDomainAnalyzerHeaderRobotsSnapshotAndCohortUtilities(t *testing.T) {
	robots := AnalyzeRobots("User-agent: *\nDisallow: /admin\nAllow: /admin/help\nSitemap: https://example.com/sitemap.xml", nil, 10)
	blocked := EvaluateRobotsPath(robots, "/admin/secret")
	if blocked.Allowed {
		t.Fatal("expected robots block")
	}
	allowed := EvaluateRobotsPath(robots, "/admin/help/article")
	if !allowed.Allowed {
		t.Fatal("specific allow should win over shorter disallow")
	}
	headers := ExtractHeaderSignals(FetchRecord{
		URL:         "https://example.com",
		ContentType: "text/html",
		Headers: map[string]string{
			"Server":                    "nginx",
			"Strict-Transport-Security": "max-age=1",
			"Content-Security-Policy":   "default-src 'self'",
			"X-Powered-By":              "Next.js",
			"Cache-Control":             "max-age=60",
		},
	})
	if headers.ServerFamily != "nginx" || headers.FrameworkHint != "nextjs" || headers.SecurityHeaderScore <= 0 {
		t.Fatalf("unexpected header signals: %#v", headers)
	}
	records := DeduplicateFetchRecords([]FetchRecord{
		{URL: "https://example.com/a?utm_source=x", StatusCode: 500, FetchedAtUnix: 1},
		{URL: "https://example.com/a", StatusCode: 200, FetchedAtUnix: 2},
		{URL: "https://example.com/b", StatusCode: 200, FetchedAtUnix: 3},
	})
	if len(records) != 2 {
		t.Fatalf("dedupe failed: %#v", records)
	}
	fp := &DomainFingerprint{
		Domain:               "example.com",
		TopologyClass:        TopologySaaSDocs,
		TopologyDistribution: map[string]float64{TopologySaaSDocs: 1},
		URLPatterns:          []URLPattern{{Pattern: "/docs/*", Count: 10, TopologyClass: TopologySaaSDocs, Confidence: 0.9}},
		RobotsSignals:        robots,
		PhaseRecommendation:  PhaseRecommendationLearning,
		FrictionLevel:        robots.FrictionLevel,
		Confidence:           0.8,
		SignalDensity:        0.7,
		ObservationCount:     20,
		FingerprintSHA256:    "hash",
	}
	snapshot := BuildDomainSnapshot(fp)
	if snapshot.Domain != "example.com" || len(snapshot.SeedPaths) == 0 {
		t.Fatalf("unexpected snapshot: %#v", snapshot)
	}
	cohort := SummarizeDomainCohort([]*DomainFingerprint{fp})
	if cohort.TotalDomains != 1 || cohort.ByTopology[TopologySaaSDocs] != 1 {
		t.Fatalf("unexpected cohort: %#v", cohort)
	}
	merged := MergeRobotsAnalysis(robots, AnalyzeRobots("User-agent: *\nDisallow: /private", nil, 10))
	if len(merged.DisallowRules) < 2 {
		t.Fatal("expected merged robots rules")
	}
}

func TestFrontierScheduler(t *testing.T) {
	s := NewFrontierScheduler(2)
	p1 := &FrontierPlan{Domain: "a.com", Priority: 0.9}
	p2 := &FrontierPlan{Domain: "b.com", Priority: 0.8}
	p3 := &FrontierPlan{Domain: "c.com", Priority: 0.95}

	s.Enqueue(p1)
	s.Enqueue(p2)
	s.Enqueue(p3)

	if s.PendingCount() != 3 {
		t.Errorf("expected 3 pending, got %d", s.PendingCount())
	}

	popped1 := s.Dequeue()
	if popped1.Domain != "c.com" {
		t.Errorf("expected highest priority c.com, got %s", popped1.Domain)
	}

	popped2 := s.Dequeue()
	if popped2.Domain != "a.com" {
		t.Errorf("expected a.com, got %s", popped2.Domain)
	}

	popped3 := s.Dequeue()
	if popped3 != nil {
		t.Errorf("expected nil due to max concurrency, got %v", popped3)
	}

	s.Complete("c.com")
	popped3 = s.Dequeue()
	if popped3 == nil || popped3.Domain != "b.com" {
		t.Errorf("expected b.com after slot freed")
	}
}

func TestAdaptiveRateLimiter(t *testing.T) {
	limiter := NewAdaptiveRateLimiter("example.com", 500)
	for i := 0; i < 5; i++ {
		limiter.RecordSuccess()
	}
	if limiter.CurrentDelayMS() >= 500 {
		t.Errorf("expected delay to decrease after 5 successes, got %d", limiter.CurrentDelayMS())
	}

	limiter.RecordFailure(429)
	if limiter.CurrentDelayMS() <= 500 {
		t.Errorf("expected delay to spike after 429, got %d", limiter.CurrentDelayMS())
	}
}

func TestZoneMerger(t *testing.T) {
	merger := DefaultZoneMerger()
	zones := []SignalZone{
		{StartByte: 0, EndByte: 100, Content: "Hello", Density: 0.8},
		{StartByte: 105, EndByte: 200, Content: "World", Density: 0.9},
		{StartByte: 1000, EndByte: 1100, Content: "Distant", Density: 0.8},
	}

	merged := merger.Merge(zones)
	if len(merged) != 2 {
		t.Errorf("expected 2 zones after merge, got %d", len(merged))
	}
	if merged[0].Content != "Hello\n\nWorld" {
		t.Errorf("expected Hello World merged, got %q", merged[0].Content)
	}
}

func TestGibberishFilter(t *testing.T) {
	if !IsGibberish("afwieuiofhwaiuefhiweuhfweafewifewaifjwaejiofawefawefawefwaefawef") {
		t.Errorf("expected long consonant string to be gibberish")
	}
	if IsGibberish("This is a completely normal sentence that has a very typical word length and structure.") {
		t.Errorf("expected normal sentence to not be gibberish")
	}
}
