package preparser

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	MinimumYieldRatio    = 0.001
	MaximumYieldRatio    = 0.30
	StaleWindowSize      = 20
	StaleThresholdFactor = 0.5
)

type RecipeHealthEvent struct {
	TopologyClass   string  `json:"topology_class"`
	RecipeHash      string  `json:"recipe_hash"`
	SampleCount     int     `json:"sample_count"`
	SuccessCount    int     `json:"success_count"`
	FailureCount    int     `json:"failure_count"`
	EmptyRate       float64 `json:"empty_rate"`
	MedianLatencyMS float64 `json:"median_latency_ms"`
	Stale           bool    `json:"stale"`
	RunID           string  `json:"run_id"`
}

type RecipeStaleEvent struct {
	TopologyClass string  `json:"topology_class"`
	RecipeHash    string  `json:"recipe_hash"`
	Reason        string  `json:"reason"`
	Confidence    float64 `json:"confidence"`
	RunID         string  `json:"run_id"`
}

type RecipeValidationSample struct {
	CleanSignal string
	LatencyMS   float64
	Succeeded   bool
}

type RecipeYieldSample struct {
	Domain         string  `json:"domain"`
	URL            string  `json:"url"`
	TopologyClass  string  `json:"topology_class"`
	RawBytes       int     `json:"raw_bytes"`
	SignalBytes    int     `json:"signal_bytes"`
	YieldRatio     float64 `json:"yield_ratio"`
	LatencyMS      float64 `json:"latency_ms"`
	Succeeded      bool    `json:"succeeded"`
	Empty          bool    `json:"empty"`
	CapturedAtUnix int64   `json:"captured_at_unix"`
}

type RecipeValidationOptions struct {
	Domain               string  `json:"domain"`
	TopologyClass        string  `json:"topology_class"`
	HistoricalYield      float64 `json:"historical_yield"`
	MinimumYieldRatio    float64 `json:"minimum_yield_ratio"`
	MaximumYieldRatio    float64 `json:"maximum_yield_ratio"`
	StaleThresholdFactor float64 `json:"stale_threshold_factor"`
	WindowSize           int     `json:"window_size"`
	NowUnix              int64   `json:"now_unix"`
	RunID                string  `json:"run_id"`
}

type RecipeValidationReport struct {
	Domain              string              `json:"domain"`
	TopologyClass       string              `json:"topology_class"`
	RecipeHash          string              `json:"recipe_hash"`
	SampleSize          int                 `json:"sample_size"`
	WindowSize          int                 `json:"window_size"`
	HistoricalYield     float64             `json:"historical_yield"`
	MeanYield           float64             `json:"mean_yield"`
	MedianYield         float64             `json:"median_yield"`
	P10Yield            float64             `json:"p10_yield"`
	P90Yield            float64             `json:"p90_yield"`
	EmptyRate           float64             `json:"empty_rate"`
	FailureRate         float64             `json:"failure_rate"`
	TooBroadRate        float64             `json:"too_broad_rate"`
	MedianLatencyMS     float64             `json:"median_latency_ms"`
	Stale               bool                `json:"stale"`
	StaleReason         string              `json:"stale_reason"`
	Confidence          float64             `json:"confidence"`
	ValidatedAtUnix     int64               `json:"validated_at_unix"`
	Samples             []RecipeYieldSample `json:"samples"`
	RecommendedAction   string              `json:"recommended_action"`
	RecommendedPriority int                 `json:"recommended_priority"`
	RunID               string              `json:"run_id"`
}

type RecipeRegistryRecord struct {
	Domain              string  `json:"domain"`
	TopologyClass       string  `json:"topology_class"`
	Recipe              string  `json:"recipe"`
	RecipeHash          string  `json:"recipe_hash"`
	HistoricalYield     float64 `json:"historical_yield"`
	HistoricalLatencyMS float64 `json:"historical_latency_ms"`
	LastValidatedUnix   int64   `json:"last_validated_unix"`
	SampleCount         int     `json:"sample_count"`
	Stale               bool    `json:"stale"`
}

type RecipeStepKind string

const (
	RecipeStepSelect       RecipeStepKind = "select"
	RecipeStepDrop         RecipeStepKind = "drop"
	RecipeStepKeepBetween  RecipeStepKind = "keep_between"
	RecipeStepStripHTML    RecipeStepKind = "strip_html"
	RecipeStepCollapseWS   RecipeStepKind = "collapse_ws"
	RecipeStepReplace      RecipeStepKind = "replace"
	RecipeStepMinTokens    RecipeStepKind = "min_tokens"
	RecipeStepMaxBytes     RecipeStepKind = "max_bytes"
	RecipeStepRequire      RecipeStepKind = "require"
	RecipeStepReject       RecipeStepKind = "reject"
	RecipeStepFirstLine    RecipeStepKind = "first_line"
	RecipeStepLastLine     RecipeStepKind = "last_line"
	RecipeStepRegexCapture RecipeStepKind = "regex_capture"
)

type RecipeStep struct {
	Kind       RecipeStepKind `json:"kind"`
	Argument   string         `json:"argument"`
	Value      string         `json:"value"`
	Line       int            `json:"line"`
	Raw        string         `json:"raw"`
	Confidence float64        `json:"confidence"`
}

type CompiledRecipe struct {
	Raw            string       `json:"raw"`
	Hash           string       `json:"hash"`
	TopologyClass  string       `json:"topology_class"`
	Steps          []RecipeStep `json:"steps"`
	CompiledAtUnix int64        `json:"compiled_at_unix"`
	Warnings       []string     `json:"warnings"`
}

type RecipeLintIssue struct {
	Line     int    `json:"line"`
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
}

type RecipeLintReport struct {
	Valid      bool              `json:"valid"`
	IssueCount int               `json:"issue_count"`
	ErrorCount int               `json:"error_count"`
	WarnCount  int               `json:"warn_count"`
	Issues     []RecipeLintIssue `json:"issues"`
	StepCount  int               `json:"step_count"`
	RecipeHash string            `json:"recipe_hash"`
}

type RecipeExecutionResult struct {
	URL                string            `json:"url"`
	TopologyClass      string            `json:"topology_class"`
	RecipeHash         string            `json:"recipe_hash"`
	Signal             string            `json:"signal"`
	RawBytes           int               `json:"raw_bytes"`
	SignalBytes        int               `json:"signal_bytes"`
	YieldRatio         float64           `json:"yield_ratio"`
	TokenCount         int               `json:"token_count"`
	StepsApplied       int               `json:"steps_applied"`
	Rejected           bool              `json:"rejected"`
	RejectReason       string            `json:"reject_reason"`
	ExecutionLatencyMS float64           `json:"execution_latency_ms"`
	Diagnostics        map[string]string `json:"diagnostics"`
}

type RecipeBatchValidation struct {
	Domain          string                   `json:"domain"`
	TopologyClass   string                   `json:"topology_class"`
	Reports         []RecipeValidationReport `json:"reports"`
	HealthEvents    []RecipeHealthEvent      `json:"health_events"`
	StaleEvents     []RecipeStaleEvent       `json:"stale_events"`
	RegistryUpdates []RecipeRegistryRecord   `json:"registry_updates"`
	StaleCount      int                      `json:"stale_count"`
	HealthyCount    int                      `json:"healthy_count"`
	GeneratedAtUnix int64                    `json:"generated_at_unix"`
}

type RecipeRegistrySnapshot struct {
	Records       []RecipeRegistryRecord `json:"records"`
	RecordCount   int                    `json:"record_count"`
	ByTopology    map[string]int         `json:"by_topology"`
	ByDomain      map[string]int         `json:"by_domain"`
	StaleCount    int                    `json:"stale_count"`
	MeanYield     float64                `json:"mean_yield"`
	MeanLatencyMS float64                `json:"mean_latency_ms"`
	SnapshotHash  string                 `json:"snapshot_hash"`
}

type RecipeTrendPoint struct {
	WindowStartUnix int64   `json:"window_start_unix"`
	WindowEndUnix   int64   `json:"window_end_unix"`
	MeanYield       float64 `json:"mean_yield"`
	MedianYield     float64 `json:"median_yield"`
	FailureRate     float64 `json:"failure_rate"`
	EmptyRate       float64 `json:"empty_rate"`
	SampleCount     int     `json:"sample_count"`
}

type RecipeTrendSummary struct {
	Domain             string             `json:"domain"`
	TopologyClass      string             `json:"topology_class"`
	RecipeHash         string             `json:"recipe_hash"`
	Points             []RecipeTrendPoint `json:"points"`
	YieldSlope         float64            `json:"yield_slope"`
	LatencySlope       float64            `json:"latency_slope"`
	StabilityScore     float64            `json:"stability_score"`
	RefreshRecommended bool               `json:"refresh_recommended"`
}

type RecipePatchSuggestion struct {
	TopologyClass string  `json:"topology_class"`
	RecipeHash    string  `json:"recipe_hash"`
	Action        string  `json:"action"`
	Step          string  `json:"step"`
	Reason        string  `json:"reason"`
	Confidence    float64 `json:"confidence"`
}

type RecipeSelectorCandidate struct {
	Selector      string  `json:"selector"`
	ZoneType      string  `json:"zone_type"`
	Support       int     `json:"support"`
	MeanDensity   float64 `json:"mean_density"`
	MeanTokens    float64 `json:"mean_tokens"`
	Confidence    float64 `json:"confidence"`
	TopologyClass string  `json:"topology_class"`
}

type RecipeMutation struct {
	Action     string  `json:"action"`
	Before     string  `json:"before"`
	After      string  `json:"after"`
	Reason     string  `json:"reason"`
	Confidence float64 `json:"confidence"`
}

type RecipeABComparison struct {
	Domain               string  `json:"domain"`
	TopologyClass        string  `json:"topology_class"`
	BaselineHash         string  `json:"baseline_hash"`
	CandidateHash        string  `json:"candidate_hash"`
	BaselineMeanYield    float64 `json:"baseline_mean_yield"`
	CandidateMeanYield   float64 `json:"candidate_mean_yield"`
	BaselineFailureRate  float64 `json:"baseline_failure_rate"`
	CandidateFailureRate float64 `json:"candidate_failure_rate"`
	Winner               string  `json:"winner"`
	Confidence           float64 `json:"confidence"`
	Reason               string  `json:"reason"`
}

type RecipeCoverageReport struct {
	Domain             string   `json:"domain"`
	TopologyClass      string   `json:"topology_class"`
	RecipeHash         string   `json:"recipe_hash"`
	URLsSeen           int      `json:"urls_seen"`
	URLsWithSignal     int      `json:"urls_with_signal"`
	URLsRejected       int      `json:"urls_rejected"`
	TotalRawBytes      int      `json:"total_raw_bytes"`
	TotalSignalBytes   int      `json:"total_signal_bytes"`
	CoverageRatio      float64  `json:"coverage_ratio"`
	YieldRatio         float64  `json:"yield_ratio"`
	RepresentativeURLs []string `json:"representative_urls"`
}

type RecipeRegistryDiff struct {
	Added          []string `json:"added"`
	Removed        []string `json:"removed"`
	Changed        []string `json:"changed"`
	Unchanged      int      `json:"unchanged"`
	RequiresReload bool     `json:"requires_reload"`
}

type RecipeSampleSplit struct {
	Train      []RecipeYieldSample `json:"train"`
	Validation []RecipeYieldSample `json:"validation"`
	Test       []RecipeYieldSample `json:"test"`
}

type RecipeCrossValidationReport struct {
	RecipeHash          string                   `json:"recipe_hash"`
	FoldReports         []RecipeValidationReport `json:"fold_reports"`
	MeanValidationYield float64                  `json:"mean_validation_yield"`
	MeanFailureRate     float64                  `json:"mean_failure_rate"`
	Stable              bool                     `json:"stable"`
	RecommendedHoldout  int                      `json:"recommended_holdout"`
}

type RecipePromotionGate struct {
	Promote       bool     `json:"promote"`
	Reasons       []string `json:"reasons"`
	Score         float64  `json:"score"`
	RequiredScore float64  `json:"required_score"`
}

type RecipeRiskReport struct {
	RecipeHash     string   `json:"recipe_hash"`
	RiskScore      float64  `json:"risk_score"`
	Risks          []string `json:"risks"`
	RequiresReview bool     `json:"requires_review"`
}

type RecipeBenchmarkStats struct {
	RecipeHash       string  `json:"recipe_hash"`
	SampleCount      int     `json:"sample_count"`
	MeanLatencyMS    float64 `json:"mean_latency_ms"`
	P95LatencyMS     float64 `json:"p95_latency_ms"`
	MeanYield        float64 `json:"mean_yield"`
	MeanTokens       float64 `json:"mean_tokens"`
	FailureRate      float64 `json:"failure_rate"`
	RejectedRate     float64 `json:"rejected_rate"`
	ThroughputPerSec float64 `json:"throughput_per_sec"`
}

type RecipeRolloutPlan struct {
	RecipeHash      string   `json:"recipe_hash"`
	Decision        string   `json:"decision"`
	CanaryPercent   int      `json:"canary_percent"`
	RequiredSamples int      `json:"required_samples"`
	Guardrails      []string `json:"guardrails"`
	Reasons         []string `json:"reasons"`
	GeneratedAtUnix int64    `json:"generated_at_unix"`
}

type RecipeFailureCluster struct {
	Reason          string   `json:"reason"`
	Count           int      `json:"count"`
	URLs            []string `json:"urls"`
	Representative  string   `json:"representative"`
	RecommendedStep string   `json:"recommended_step"`
}

type RecipeRepairPlan struct {
	RecipeHash      string                  `json:"recipe_hash"`
	Actions         []RecipePatchSuggestion `json:"actions"`
	Mutations       []RecipeMutation        `json:"mutations"`
	FailureClusters []RecipeFailureCluster  `json:"failure_clusters"`
	Priority        int                     `json:"priority"`
	Confidence      float64                 `json:"confidence"`
}

type RecipeQualityScorecard struct {
	RecipeHash     string  `json:"recipe_hash"`
	YieldScore     float64 `json:"yield_score"`
	CoverageScore  float64 `json:"coverage_score"`
	LatencyScore   float64 `json:"latency_score"`
	StabilityScore float64 `json:"stability_score"`
	RiskPenalty    float64 `json:"risk_penalty"`
	OverallScore   float64 `json:"overall_score"`
	Grade          string  `json:"grade"`
}

type RecipeRevalidationCandidate struct {
	Record   RecipeRegistryRecord `json:"record"`
	Priority float64              `json:"priority"`
	Reason   string               `json:"reason"`
}

type RecipeManifestEntry struct {
	Domain          string  `json:"domain"`
	TopologyClass   string  `json:"topology_class"`
	RecipeHash      string  `json:"recipe_hash"`
	Stale           bool    `json:"stale"`
	HistoricalYield float64 `json:"historical_yield"`
	SampleCount     int     `json:"sample_count"`
}

type RecipeManifest struct {
	Entries         []RecipeManifestEntry `json:"entries"`
	GeneratedAtUnix int64                 `json:"generated_at_unix"`
	ManifestHash    string                `json:"manifest_hash"`
}

type RecipeCompatibilityIssue struct {
	RecipeHash string `json:"recipe_hash"`
	Code       string `json:"code"`
	Severity   string `json:"severity"`
	Message    string `json:"message"`
}

type RecipeCompatibilityReport struct {
	Compatible bool                       `json:"compatible"`
	Issues     []RecipeCompatibilityIssue `json:"issues"`
	ErrorCount int                        `json:"error_count"`
	WarnCount  int                        `json:"warn_count"`
}

type RecipeMigrationStep struct {
	Action        string `json:"action"`
	Domain        string `json:"domain"`
	TopologyClass string `json:"topology_class"`
	RecipeHash    string `json:"recipe_hash"`
	Reason        string `json:"reason"`
}

type RecipeMigrationPlan struct {
	Steps           []RecipeMigrationStep `json:"steps"`
	RequiresBackup  bool                  `json:"requires_backup"`
	GeneratedAtUnix int64                 `json:"generated_at_unix"`
	PlanHash        string                `json:"plan_hash"`
}

type RecipeManifestDiff struct {
	Added           []RecipeManifestEntry `json:"added"`
	Removed         []RecipeManifestEntry `json:"removed"`
	Changed         []RecipeManifestEntry `json:"changed"`
	Unchanged       int                   `json:"unchanged"`
	RequiresRestart bool                  `json:"requires_restart"`
}

type RecipeAuditEntry struct {
	UnixTime      int64  `json:"unix_time"`
	RecipeHash    string `json:"recipe_hash"`
	Action        string `json:"action"`
	Actor         string `json:"actor"`
	Reason        string `json:"reason"`
	TopologyClass string `json:"topology_class"`
}

type RecipeAuditTrail struct {
	Entries []RecipeAuditEntry `json:"entries"`
	Digest  string             `json:"digest"`
}

type RecipeInvariantSummary struct {
	RecordCount       int      `json:"record_count"`
	HashMismatches    int      `json:"hash_mismatches"`
	EmptyRecipes      int      `json:"empty_recipes"`
	MissingDomains    int      `json:"missing_domains"`
	MissingTopologies int      `json:"missing_topologies"`
	Warnings          []string `json:"warnings"`
	Healthy           bool     `json:"healthy"`
}

func ValidateRecipe(topologyClass string, recipe string, samples []RecipeValidationSample, runID string) (RecipeHealthEvent, *RecipeStaleEvent, error) {
	if topologyClass == "" {
		return RecipeHealthEvent{}, nil, errors.New("topology_class is empty")
	}
	if recipe == "" {
		return RecipeHealthEvent{}, nil, errors.New("recipe is empty")
	}
	if runID == "" {
		return RecipeHealthEvent{}, nil, errors.New("run_id is empty")
	}
	success, failure, empty := 0, 0, 0
	latencies := make([]float64, 0, len(samples))
	for _, s := range samples {
		if s.Succeeded {
			success++
		} else {
			failure++
		}
		if strings.TrimSpace(s.CleanSignal) == "" {
			empty++
		}
		if s.LatencyMS >= 0 {
			latencies = append(latencies, s.LatencyMS)
		}
	}
	emptyRate := 0.0
	if len(samples) > 0 {
		emptyRate = float64(empty) / float64(len(samples))
	}
	stale := emptyRate >= 0.4 || failure > success
	health := RecipeHealthEvent{
		TopologyClass:   topologyClass,
		RecipeHash:      hashRecipe(recipe),
		SampleCount:     len(samples),
		SuccessCount:    success,
		FailureCount:    failure,
		EmptyRate:       emptyRate,
		MedianLatencyMS: median(latencies),
		Stale:           stale,
		RunID:           runID,
	}
	if stale {
		reason := "empty_rate_high"
		if failure > success {
			reason = "failure_rate_high"
		}
		staleEvent := &RecipeStaleEvent{TopologyClass: topologyClass, RecipeHash: health.RecipeHash, Reason: reason, Confidence: staleConfidence(health), RunID: runID}
		return health, staleEvent, nil
	}
	return health, nil, nil
}

func ValidateRecipeWindow(record RecipeRegistryRecord, samples []RecipeYieldSample, opts RecipeValidationOptions) (RecipeValidationReport, RecipeHealthEvent, *RecipeStaleEvent, error) {
	if record.TopologyClass == "" && opts.TopologyClass == "" {
		return RecipeValidationReport{}, RecipeHealthEvent{}, nil, errors.New("topology_class is empty")
	}
	if record.Recipe == "" {
		return RecipeValidationReport{}, RecipeHealthEvent{}, nil, errors.New("recipe is empty")
	}
	if opts.RunID == "" {
		return RecipeValidationReport{}, RecipeHealthEvent{}, nil, errors.New("run_id is empty")
	}
	opts = normalizeValidationOptions(record, opts)
	window := selectValidationWindow(samples, opts.WindowSize)
	report := BuildRecipeValidationReport(record, window, opts)
	health := RecipeHealthEvent{
		TopologyClass:   opts.TopologyClass,
		RecipeHash:      report.RecipeHash,
		SampleCount:     report.SampleSize,
		SuccessCount:    report.SampleSize - int(math.Round(report.FailureRate*float64(report.SampleSize))),
		FailureCount:    int(math.Round(report.FailureRate * float64(report.SampleSize))),
		EmptyRate:       report.EmptyRate,
		MedianLatencyMS: report.MedianLatencyMS,
		Stale:           report.Stale,
		RunID:           opts.RunID,
	}
	if report.Stale {
		stale := &RecipeStaleEvent{
			TopologyClass: opts.TopologyClass,
			RecipeHash:    report.RecipeHash,
			Reason:        report.StaleReason,
			Confidence:    report.Confidence,
			RunID:         opts.RunID,
		}
		return report, health, stale, nil
	}
	return report, health, nil, nil
}

func BuildRecipeValidationReport(record RecipeRegistryRecord, samples []RecipeYieldSample, opts RecipeValidationOptions) RecipeValidationReport {
	yields := make([]float64, 0, len(samples))
	latencies := make([]float64, 0, len(samples))
	empty := 0
	failures := 0
	tooBroad := 0
	for i := range samples {
		samples[i] = normalizeYieldSample(samples[i])
		if samples[i].YieldRatio < 0 {
			samples[i].YieldRatio = 0
		}
		yields = append(yields, samples[i].YieldRatio)
		if samples[i].LatencyMS >= 0 {
			latencies = append(latencies, samples[i].LatencyMS)
		}
		if samples[i].Empty || samples[i].SignalBytes == 0 {
			empty++
		}
		if !samples[i].Succeeded {
			failures++
		}
		if samples[i].YieldRatio > opts.MaximumYieldRatio {
			tooBroad++
		}
	}
	sort.Float64s(yields)
	sort.Float64s(latencies)
	mean := meanFloat64(yields)
	medianYield := percentileFloat64(yields, 0.50)
	emptyRate := ratioInt(empty, len(samples))
	failureRate := ratioInt(failures, len(samples))
	tooBroadRate := ratioInt(tooBroad, len(samples))
	stale, reason := recipeStalenessDecision(mean, medianYield, emptyRate, failureRate, tooBroadRate, opts)
	confidence := recipeValidationConfidence(len(samples), mean, medianYield, emptyRate, failureRate, tooBroadRate, opts)
	action, priority := recommendedRecipeAction(stale, reason, confidence)
	return RecipeValidationReport{
		Domain:              opts.Domain,
		TopologyClass:       opts.TopologyClass,
		RecipeHash:          hashRecipe(record.Recipe),
		SampleSize:          len(samples),
		WindowSize:          opts.WindowSize,
		HistoricalYield:     opts.HistoricalYield,
		MeanYield:           mean,
		MedianYield:         medianYield,
		P10Yield:            percentileFloat64(yields, 0.10),
		P90Yield:            percentileFloat64(yields, 0.90),
		EmptyRate:           emptyRate,
		FailureRate:         failureRate,
		TooBroadRate:        tooBroadRate,
		MedianLatencyMS:     percentileFloat64(latencies, 0.50),
		Stale:               stale,
		StaleReason:         reason,
		Confidence:          confidence,
		ValidatedAtUnix:     opts.NowUnix,
		Samples:             append([]RecipeYieldSample(nil), samples...),
		RecommendedAction:   action,
		RecommendedPriority: priority,
		RunID:               opts.RunID,
	}
}

func EvaluateRecipeAgainstSignals(record RecipeRegistryRecord, rawSamples map[string]string, extracted map[string]string, opts RecipeValidationOptions) (RecipeValidationReport, error) {
	if len(rawSamples) == 0 {
		return RecipeValidationReport{}, errors.New("raw sample set is empty")
	}
	samples := make([]RecipeYieldSample, 0, len(rawSamples))
	now := opts.NowUnix
	if now <= 0 {
		now = time.Now().Unix()
	}
	keys := make([]string, 0, len(rawSamples))
	for key := range rawSamples {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, sampleURL := range keys {
		raw := rawSamples[sampleURL]
		signal := extracted[sampleURL]
		rawBytes := len([]byte(raw))
		signalBytes := len([]byte(strings.TrimSpace(signal)))
		yield := 0.0
		if rawBytes > 0 {
			yield = float64(signalBytes) / float64(rawBytes)
		}
		samples = append(samples, RecipeYieldSample{
			Domain:         opts.Domain,
			URL:            sampleURL,
			TopologyClass:  opts.TopologyClass,
			RawBytes:       rawBytes,
			SignalBytes:    signalBytes,
			YieldRatio:     yield,
			Succeeded:      signalBytes > 0,
			Empty:          signalBytes == 0,
			CapturedAtUnix: now,
		})
	}
	report, _, _, err := ValidateRecipeWindow(record, samples, opts)
	return report, err
}

func ApplyValidationToRegistry(record RecipeRegistryRecord, report RecipeValidationReport) RecipeRegistryRecord {
	if record.RecipeHash == "" {
		record.RecipeHash = hashRecipe(record.Recipe)
	}
	record.Domain = firstNonEmpty(record.Domain, report.Domain)
	record.TopologyClass = firstNonEmpty(record.TopologyClass, report.TopologyClass)
	record.HistoricalYield = updateHistoricalYield(record.HistoricalYield, report.MeanYield, record.SampleCount, report.SampleSize)
	record.HistoricalLatencyMS = updateHistoricalYield(record.HistoricalLatencyMS, report.MedianLatencyMS, record.SampleCount, report.SampleSize)
	record.LastValidatedUnix = report.ValidatedAtUnix
	record.SampleCount += report.SampleSize
	record.Stale = report.Stale
	return record
}

func CompileRecipe(topologyClass string, recipe string, nowUnix int64) (CompiledRecipe, RecipeLintReport) {
	if nowUnix <= 0 {
		nowUnix = time.Now().Unix()
	}
	steps, parseIssues := ParseRecipeSteps(recipe)
	report := LintRecipeSteps(steps, parseIssues)
	compiled := CompiledRecipe{
		Raw:            recipe,
		Hash:           hashRecipe(recipe),
		TopologyClass:  topologyClass,
		Steps:          steps,
		CompiledAtUnix: nowUnix,
	}
	for _, issue := range report.Issues {
		if issue.Severity == "warning" {
			compiled.Warnings = append(compiled.Warnings, issue.Message)
		}
	}
	return compiled, report
}

func ParseRecipeSteps(recipe string) ([]RecipeStep, []RecipeLintIssue) {
	parts := splitRecipeStatements(recipe)
	steps := make([]RecipeStep, 0, len(parts))
	issues := make([]RecipeLintIssue, 0)
	for idx, raw := range parts {
		line := idx + 1
		stmt := strings.TrimSpace(raw)
		if stmt == "" || strings.HasPrefix(stmt, "#") {
			continue
		}
		step, err := parseRecipeStatement(stmt, line)
		if err != nil {
			issues = append(issues, RecipeLintIssue{Line: line, Code: "parse_error", Severity: "error", Message: err.Error()})
			continue
		}
		steps = append(steps, step)
	}
	return steps, issues
}

func LintRecipeSteps(steps []RecipeStep, issues []RecipeLintIssue) RecipeLintReport {
	report := RecipeLintReport{Valid: true, Issues: append([]RecipeLintIssue(nil), issues...), StepCount: len(steps)}
	seenSelectors := make(map[string]int)
	hasExtractor := false
	for _, step := range steps {
		if step.Kind == "" {
			report.Issues = append(report.Issues, RecipeLintIssue{Line: step.Line, Code: "empty_kind", Severity: "error", Message: "recipe step kind is empty"})
			continue
		}
		if !recipeStepKindKnown(step.Kind) {
			report.Issues = append(report.Issues, RecipeLintIssue{Line: step.Line, Code: "unknown_step", Severity: "error", Message: "unknown recipe step: " + string(step.Kind)})
		}
		if step.Kind == RecipeStepSelect || step.Kind == RecipeStepKeepBetween || step.Kind == RecipeStepRegexCapture {
			hasExtractor = true
		}
		switch step.Kind {
		case RecipeStepSelect, RecipeStepDrop, RecipeStepRequire, RecipeStepReject, RecipeStepRegexCapture:
			if strings.TrimSpace(step.Argument) == "" {
				report.Issues = append(report.Issues, RecipeLintIssue{Line: step.Line, Code: "missing_argument", Severity: "error", Message: string(step.Kind) + " requires an argument"})
			}
		case RecipeStepKeepBetween, RecipeStepReplace:
			if strings.TrimSpace(step.Argument) == "" || strings.TrimSpace(step.Value) == "" {
				report.Issues = append(report.Issues, RecipeLintIssue{Line: step.Line, Code: "missing_pair", Severity: "error", Message: string(step.Kind) + " requires argument and value"})
			}
		case RecipeStepMinTokens, RecipeStepMaxBytes:
			if _, err := strconv.Atoi(strings.TrimSpace(step.Argument)); err != nil {
				report.Issues = append(report.Issues, RecipeLintIssue{Line: step.Line, Code: "invalid_number", Severity: "error", Message: string(step.Kind) + " requires an integer"})
			}
		}
		if step.Kind == RecipeStepSelect {
			seenSelectors[step.Argument]++
			if seenSelectors[step.Argument] > 1 {
				report.Issues = append(report.Issues, RecipeLintIssue{Line: step.Line, Code: "duplicate_selector", Severity: "warning", Message: "selector appears more than once: " + step.Argument})
			}
		}
	}
	if len(steps) == 0 {
		report.Issues = append(report.Issues, RecipeLintIssue{Line: 0, Code: "empty_recipe", Severity: "error", Message: "recipe has no executable steps"})
	}
	if !hasExtractor {
		report.Issues = append(report.Issues, RecipeLintIssue{Line: 0, Code: "no_extractor", Severity: "warning", Message: "recipe has no explicit extraction step"})
	}
	for _, issue := range report.Issues {
		report.IssueCount++
		if issue.Severity == "error" {
			report.ErrorCount++
			report.Valid = false
		} else {
			report.WarnCount++
		}
	}
	report.RecipeHash = hashRecipe(renderRecipeSteps(steps))
	return report
}

func ExecuteCompiledRecipe(compiled CompiledRecipe, raw string, sampleURL string) RecipeExecutionResult {
	start := time.Now()
	current := raw
	result := RecipeExecutionResult{
		URL:           sampleURL,
		TopologyClass: compiled.TopologyClass,
		RecipeHash:    compiled.Hash,
		RawBytes:      len([]byte(raw)),
		Diagnostics:   map[string]string{},
	}
	for _, step := range compiled.Steps {
		before := current
		next, rejected, reason := executeRecipeStep(step, current)
		result.StepsApplied++
		if rejected {
			result.Rejected = true
			result.RejectReason = reason
			current = ""
			break
		}
		current = next
		if before != current {
			result.Diagnostics["last_changed_by"] = string(step.Kind)
		}
	}
	result.Signal = strings.TrimSpace(current)
	result.SignalBytes = len([]byte(result.Signal))
	result.TokenCount = countTokens(result.Signal)
	if result.RawBytes > 0 {
		result.YieldRatio = float64(result.SignalBytes) / float64(result.RawBytes)
	}
	result.ExecutionLatencyMS = float64(time.Since(start).Microseconds()) / 1000.0
	return result
}

func EvaluateCompiledRecipe(record RecipeRegistryRecord, rawSamples map[string]string, opts RecipeValidationOptions) (RecipeValidationReport, []RecipeExecutionResult, error) {
	compiled, lint := CompileRecipe(firstNonEmpty(record.TopologyClass, opts.TopologyClass), record.Recipe, opts.NowUnix)
	if !lint.Valid {
		return RecipeValidationReport{}, nil, errors.New("recipe lint failed")
	}
	results := make([]RecipeExecutionResult, 0, len(rawSamples))
	yieldSamples := make([]RecipeYieldSample, 0, len(rawSamples))
	keys := make([]string, 0, len(rawSamples))
	for key := range rawSamples {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	now := opts.NowUnix
	if now <= 0 {
		now = time.Now().Unix()
	}
	for _, sampleURL := range keys {
		exec := ExecuteCompiledRecipe(compiled, rawSamples[sampleURL], sampleURL)
		results = append(results, exec)
		yieldSamples = append(yieldSamples, RecipeYieldSample{
			Domain:         opts.Domain,
			URL:            sampleURL,
			TopologyClass:  compiled.TopologyClass,
			RawBytes:       exec.RawBytes,
			SignalBytes:    exec.SignalBytes,
			YieldRatio:     exec.YieldRatio,
			LatencyMS:      exec.ExecutionLatencyMS,
			Succeeded:      !exec.Rejected && exec.SignalBytes > 0,
			Empty:          exec.SignalBytes == 0,
			CapturedAtUnix: now,
		})
	}
	report, _, _, err := ValidateRecipeWindow(record, yieldSamples, opts)
	return report, results, err
}

func ValidateRecipeRegistry(records []RecipeRegistryRecord) RecipeRegistrySnapshot {
	snapshot := RecipeRegistrySnapshot{
		Records:     append([]RecipeRegistryRecord(nil), records...),
		RecordCount: len(records),
		ByTopology:  make(map[string]int),
		ByDomain:    make(map[string]int),
	}
	yieldSum := 0.0
	latencySum := 0.0
	for _, record := range records {
		snapshot.ByTopology[record.TopologyClass]++
		snapshot.ByDomain[record.Domain]++
		if record.Stale {
			snapshot.StaleCount++
		}
		yieldSum += record.HistoricalYield
		latencySum += record.HistoricalLatencyMS
	}
	if len(records) > 0 {
		snapshot.MeanYield = yieldSum / float64(len(records))
		snapshot.MeanLatencyMS = latencySum / float64(len(records))
	}
	payload, _ := json.Marshal(snapshot.Records)
	snapshot.SnapshotHash = hashRecipe(string(payload))
	return snapshot
}

func ValidateRegistryBatch(records []RecipeRegistryRecord, sampleSets map[string][]RecipeYieldSample, opts RecipeValidationOptions) RecipeBatchValidation {
	batch := RecipeBatchValidation{
		Domain:          opts.Domain,
		TopologyClass:   opts.TopologyClass,
		GeneratedAtUnix: time.Now().Unix(),
	}
	for _, record := range records {
		key := recipeRegistryKey(record)
		localOpts := opts
		if localOpts.Domain == "" {
			localOpts.Domain = record.Domain
		}
		if localOpts.TopologyClass == "" {
			localOpts.TopologyClass = record.TopologyClass
		}
		if localOpts.RunID == "" {
			localOpts.RunID = deterministicID(record.Recipe + key)
		}
		report, health, stale, err := ValidateRecipeWindow(record, sampleSets[key], localOpts)
		if err != nil {
			continue
		}
		batch.Reports = append(batch.Reports, report)
		batch.HealthEvents = append(batch.HealthEvents, health)
		updated := ApplyValidationToRegistry(record, report)
		batch.RegistryUpdates = append(batch.RegistryUpdates, updated)
		if stale != nil {
			batch.StaleEvents = append(batch.StaleEvents, *stale)
			batch.StaleCount++
		} else {
			batch.HealthyCount++
		}
	}
	return batch
}

func BuildRecipeTrend(record RecipeRegistryRecord, samples []RecipeYieldSample, windowSeconds int64) RecipeTrendSummary {
	if windowSeconds <= 0 {
		windowSeconds = 86400
	}
	cp := append([]RecipeYieldSample(nil), samples...)
	sort.Slice(cp, func(i, j int) bool {
		if cp[i].CapturedAtUnix != cp[j].CapturedAtUnix {
			return cp[i].CapturedAtUnix < cp[j].CapturedAtUnix
		}
		return cp[i].URL < cp[j].URL
	})
	summary := RecipeTrendSummary{Domain: record.Domain, TopologyClass: record.TopologyClass, RecipeHash: hashRecipe(record.Recipe)}
	if len(cp) == 0 {
		summary.RefreshRecommended = true
		return summary
	}
	windowStart := cp[0].CapturedAtUnix
	var bucket []RecipeYieldSample
	flush := func(end int64) {
		if len(bucket) == 0 {
			return
		}
		point := buildTrendPoint(windowStart, end, bucket)
		summary.Points = append(summary.Points, point)
		bucket = nil
	}
	for _, sample := range cp {
		if sample.CapturedAtUnix-windowStart >= windowSeconds {
			flush(sample.CapturedAtUnix)
			windowStart = sample.CapturedAtUnix
		}
		bucket = append(bucket, sample)
	}
	flush(windowStart + windowSeconds)
	summary.YieldSlope = trendSlope(summary.Points, func(p RecipeTrendPoint) float64 { return p.MeanYield })
	summary.LatencySlope = trendSlope(summary.Points, func(p RecipeTrendPoint) float64 { return p.FailureRate })
	summary.StabilityScore = recipeTrendStability(summary.Points)
	summary.RefreshRecommended = summary.YieldSlope < -MinimumYieldRatio || summary.StabilityScore < 0.4
	return summary
}

func SuggestRecipePatches(report RecipeValidationReport, lint RecipeLintReport) []RecipePatchSuggestion {
	suggestions := make([]RecipePatchSuggestion, 0)
	add := func(action, step, reason string, confidence float64) {
		suggestions = append(suggestions, RecipePatchSuggestion{
			TopologyClass: report.TopologyClass,
			RecipeHash:    report.RecipeHash,
			Action:        action,
			Step:          step,
			Reason:        reason,
			Confidence:    clampFloat(confidence, 0, 1),
		})
	}
	if lint.ErrorCount > 0 {
		add("fix_syntax", "", "recipe lint has errors", 0.95)
	}
	if report.EmptyRate >= 0.4 {
		add("broaden_selector", "select:main, article, [data-content]", "empty rate is high", report.EmptyRate)
	}
	if report.TooBroadRate >= 0.25 || report.MeanYield > MaximumYieldRatio {
		add("tighten_selector", "drop:nav; drop:footer; drop:aside", "yield is too broad", math.Max(report.TooBroadRate, report.MeanYield))
	}
	if report.FailureRate >= 0.5 {
		add("add_required_marker", "require:<title", "recipe failure rate is high", report.FailureRate)
	}
	if report.MedianLatencyMS > 500 {
		add("simplify_recipe", "strip_html; collapse_ws", "recipe is slow", clampFloat(report.MedianLatencyMS/2000, 0, 1))
	}
	sort.Slice(suggestions, func(i, j int) bool {
		if suggestions[i].Confidence != suggestions[j].Confidence {
			return suggestions[i].Confidence > suggestions[j].Confidence
		}
		return suggestions[i].Action < suggestions[j].Action
	})
	return suggestions
}

func MineSelectorCandidates(topologyClass string, zones []SignalZone) []RecipeSelectorCandidate {
	type acc struct {
		support  int
		density  float64
		tokens   float64
		zoneType string
	}
	bySelector := make(map[string]*acc)
	for _, zone := range zones {
		selectors := candidateSelectorsForZone(zone)
		for _, selector := range selectors {
			a := bySelector[selector]
			if a == nil {
				a = &acc{zoneType: string(zone.Type)}
				bySelector[selector] = a
			}
			a.support++
			a.density += zone.Density
			a.tokens += float64(zone.TokenCount)
		}
	}
	out := make([]RecipeSelectorCandidate, 0, len(bySelector))
	for selector, a := range bySelector {
		meanDensity := a.density / float64(a.support)
		meanTokens := a.tokens / float64(a.support)
		confidence := clampFloat(meanDensity*0.45+math.Min(1, meanTokens/250.0)*0.35+math.Min(1, float64(a.support)/10.0)*0.20, 0, 1)
		out = append(out, RecipeSelectorCandidate{
			Selector:      selector,
			ZoneType:      a.zoneType,
			Support:       a.support,
			MeanDensity:   meanDensity,
			MeanTokens:    meanTokens,
			Confidence:    confidence,
			TopologyClass: topologyClass,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Confidence != out[j].Confidence {
			return out[i].Confidence > out[j].Confidence
		}
		if out[i].Support != out[j].Support {
			return out[i].Support > out[j].Support
		}
		return out[i].Selector < out[j].Selector
	})
	return out
}

func DraftRecipeFromZones(topologyClass string, zones []SignalZone, maxSteps int) string {
	if maxSteps <= 0 {
		maxSteps = 8
	}
	candidates := MineSelectorCandidates(topologyClass, zones)
	var steps []string
	for _, candidate := range candidates {
		if len(steps) >= maxSteps {
			break
		}
		if candidate.Confidence < 0.25 {
			continue
		}
		steps = append(steps, "select:"+candidate.Selector)
	}
	if len(steps) == 0 {
		steps = append(steps, "select:main", "select:article")
	}
	steps = append(steps, "drop:nav", "drop:footer", "strip_html", "collapse_ws", "min_tokens:5")
	return strings.Join(steps, ";\n")
}

func MutateRecipe(recipe string, report RecipeValidationReport) []RecipeMutation {
	mutations := make([]RecipeMutation, 0)
	add := func(action, after, reason string, confidence float64) {
		mutations = append(mutations, RecipeMutation{
			Action:     action,
			Before:     recipe,
			After:      after,
			Reason:     reason,
			Confidence: clampFloat(confidence, 0, 1),
		})
	}
	if report.EmptyRate >= 0.35 {
		add("broaden", recipe+";\nselect:main;\nselect:article", "empty rate indicates selector miss", report.EmptyRate)
	}
	if report.TooBroadRate >= 0.20 || report.MeanYield > MaximumYieldRatio {
		add("tighten", recipe+";\ndrop:nav;\ndrop:footer;\ndrop:aside", "yield too broad", math.Max(report.TooBroadRate, report.MeanYield))
	}
	if report.FailureRate >= 0.25 {
		add("guard", recipe+";\nrequire:<", "failures need structural guard", report.FailureRate)
	}
	if report.MedianLatencyMS > 500 {
		add("simplify", "strip_html;\ncollapse_ws;\nmin_tokens:5", "latency too high", clampFloat(report.MedianLatencyMS/2000.0, 0, 1))
	}
	sort.Slice(mutations, func(i, j int) bool {
		if mutations[i].Confidence != mutations[j].Confidence {
			return mutations[i].Confidence > mutations[j].Confidence
		}
		return mutations[i].Action < mutations[j].Action
	})
	return mutations
}

func ApplyRecipeMutation(mutation RecipeMutation) (CompiledRecipe, RecipeLintReport) {
	return CompileRecipe("", mutation.After, time.Now().Unix())
}

func CompareRecipesAB(record RecipeRegistryRecord, baseline string, candidate string, rawSamples map[string]string, opts RecipeValidationOptions) RecipeABComparison {
	baseRecord := record
	baseRecord.Recipe = baseline
	candRecord := record
	candRecord.Recipe = candidate
	baseReport, _, _ := EvaluateCompiledRecipe(baseRecord, rawSamples, opts)
	candReport, _, _ := EvaluateCompiledRecipe(candRecord, rawSamples, opts)
	comparison := RecipeABComparison{
		Domain:               firstNonEmpty(record.Domain, opts.Domain),
		TopologyClass:        firstNonEmpty(record.TopologyClass, opts.TopologyClass),
		BaselineHash:         hashRecipe(baseline),
		CandidateHash:        hashRecipe(candidate),
		BaselineMeanYield:    baseReport.MeanYield,
		CandidateMeanYield:   candReport.MeanYield,
		BaselineFailureRate:  baseReport.FailureRate,
		CandidateFailureRate: candReport.FailureRate,
	}
	baseScore := recipeABScore(baseReport)
	candScore := recipeABScore(candReport)
	if candScore > baseScore {
		comparison.Winner = "candidate"
		comparison.Confidence = clampFloat(candScore-baseScore, 0, 1)
		comparison.Reason = "candidate_score_higher"
	} else {
		comparison.Winner = "baseline"
		comparison.Confidence = clampFloat(baseScore-candScore, 0, 1)
		comparison.Reason = "baseline_score_higher"
	}
	return comparison
}

func ComputeRecipeCoverage(results []RecipeExecutionResult, domain string, topologyClass string, recipeHash string) RecipeCoverageReport {
	report := RecipeCoverageReport{Domain: domain, TopologyClass: topologyClass, RecipeHash: recipeHash}
	for _, result := range results {
		report.URLsSeen++
		report.TotalRawBytes += result.RawBytes
		report.TotalSignalBytes += result.SignalBytes
		if result.SignalBytes > 0 && !result.Rejected {
			report.URLsWithSignal++
			if len(report.RepresentativeURLs) < 10 {
				report.RepresentativeURLs = append(report.RepresentativeURLs, result.URL)
			}
		}
		if result.Rejected {
			report.URLsRejected++
		}
	}
	report.CoverageRatio = ratioInt(report.URLsWithSignal, report.URLsSeen)
	if report.TotalRawBytes > 0 {
		report.YieldRatio = float64(report.TotalSignalBytes) / float64(report.TotalRawBytes)
	}
	return report
}

func SplitRecipeSamples(samples []RecipeYieldSample, validationRatio float64, testRatio float64) RecipeSampleSplit {
	if validationRatio <= 0 || validationRatio >= 1 {
		validationRatio = 0.20
	}
	if testRatio < 0 || testRatio >= 1 {
		testRatio = 0.10
	}
	cp := append([]RecipeYieldSample(nil), samples...)
	sort.Slice(cp, func(i, j int) bool {
		if cp[i].CapturedAtUnix != cp[j].CapturedAtUnix {
			return cp[i].CapturedAtUnix < cp[j].CapturedAtUnix
		}
		return cp[i].URL < cp[j].URL
	})
	n := len(cp)
	testN := int(math.Round(float64(n) * testRatio))
	valN := int(math.Round(float64(n) * validationRatio))
	if testN+valN > n {
		valN = maxInt(0, n-testN)
	}
	trainN := n - valN - testN
	return RecipeSampleSplit{
		Train:      append([]RecipeYieldSample(nil), cp[:trainN]...),
		Validation: append([]RecipeYieldSample(nil), cp[trainN:trainN+valN]...),
		Test:       append([]RecipeYieldSample(nil), cp[trainN+valN:]...),
	}
}

func CrossValidateRecipe(record RecipeRegistryRecord, samples []RecipeYieldSample, folds int, opts RecipeValidationOptions) RecipeCrossValidationReport {
	if folds <= 1 {
		folds = 3
	}
	if folds > len(samples) && len(samples) > 0 {
		folds = len(samples)
	}
	report := RecipeCrossValidationReport{RecipeHash: hashRecipe(record.Recipe), RecommendedHoldout: 1}
	if len(samples) == 0 {
		report.Stable = false
		return report
	}
	for fold := 0; fold < folds; fold++ {
		var validation []RecipeYieldSample
		for i, sample := range samples {
			if i%folds == fold {
				validation = append(validation, sample)
			}
		}
		localOpts := opts
		if localOpts.RunID == "" {
			localOpts.RunID = deterministicID(record.Recipe + strconv.Itoa(fold))
		}
		foldReport := BuildRecipeValidationReport(record, validation, normalizeValidationOptions(record, localOpts))
		report.FoldReports = append(report.FoldReports, foldReport)
		report.MeanValidationYield += foldReport.MeanYield
		report.MeanFailureRate += foldReport.FailureRate
	}
	if len(report.FoldReports) > 0 {
		report.MeanValidationYield /= float64(len(report.FoldReports))
		report.MeanFailureRate /= float64(len(report.FoldReports))
	}
	report.Stable = report.MeanValidationYield >= MinimumYieldRatio && report.MeanFailureRate < 0.35 && crossValidationYieldVariance(report.FoldReports) < 0.01
	report.RecommendedHoldout = maxInt(1, len(samples)/maxInt(5, folds))
	return report
}

func DiffRecipeRegistries(oldRecords []RecipeRegistryRecord, newRecords []RecipeRegistryRecord) RecipeRegistryDiff {
	oldMap := map[string]RecipeRegistryRecord{}
	newMap := map[string]RecipeRegistryRecord{}
	for _, record := range oldRecords {
		oldMap[recipeRegistryIdentity(record)] = record
	}
	for _, record := range newRecords {
		newMap[recipeRegistryIdentity(record)] = record
	}
	diff := RecipeRegistryDiff{}
	for key, record := range newMap {
		old, ok := oldMap[key]
		if !ok {
			diff.Added = append(diff.Added, key)
			continue
		}
		if old.RecipeHash != record.RecipeHash || old.Stale != record.Stale || math.Abs(old.HistoricalYield-record.HistoricalYield) > 0.001 {
			diff.Changed = append(diff.Changed, key)
		} else {
			diff.Unchanged++
		}
	}
	for key := range oldMap {
		if _, ok := newMap[key]; !ok {
			diff.Removed = append(diff.Removed, key)
		}
	}
	sort.Strings(diff.Added)
	sort.Strings(diff.Removed)
	sort.Strings(diff.Changed)
	diff.RequiresReload = len(diff.Added) > 0 || len(diff.Removed) > 0 || len(diff.Changed) > 0
	return diff
}

func NormalizeRegistryRecords(records []RecipeRegistryRecord) []RecipeRegistryRecord {
	out := append([]RecipeRegistryRecord(nil), records...)
	for i := range out {
		out[i].Domain = normalizeDomain(out[i].Domain)
		out[i].TopologyClass = strings.TrimSpace(out[i].TopologyClass)
		if out[i].RecipeHash == "" && out[i].Recipe != "" {
			out[i].RecipeHash = hashRecipe(out[i].Recipe)
		}
		if out[i].HistoricalYield < 0 {
			out[i].HistoricalYield = 0
		}
		if out[i].HistoricalLatencyMS < 0 {
			out[i].HistoricalLatencyMS = 0
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return recipeRegistryIdentity(out[i]) < recipeRegistryIdentity(out[j])
	})
	return out
}

func SerializeRecipeRegistry(records []RecipeRegistryRecord) ([]byte, error) {
	normalized := NormalizeRegistryRecords(records)
	return json.Marshal(normalized)
}

func DeserializeRecipeRegistry(data []byte) ([]RecipeRegistryRecord, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var records []RecipeRegistryRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, err
	}
	return NormalizeRegistryRecords(records), nil
}

func GateRecipePromotion(report RecipeValidationReport, lint RecipeLintReport, coverage RecipeCoverageReport, requiredScore float64) RecipePromotionGate {
	if requiredScore <= 0 {
		requiredScore = 0.70
	}
	gate := RecipePromotionGate{Promote: true, RequiredScore: requiredScore}
	score := 0.0
	score += clampFloat(report.MeanYield/MaximumYieldRatio, 0, 1) * 0.25
	score += (1 - report.FailureRate) * 0.25
	score += (1 - report.EmptyRate) * 0.20
	score += coverage.CoverageRatio * 0.20
	if lint.Valid {
		score += 0.10
	} else {
		gate.Reasons = append(gate.Reasons, "lint_failed")
	}
	if report.Stale {
		gate.Reasons = append(gate.Reasons, "report_marked_stale")
	}
	if coverage.URLsSeen == 0 {
		gate.Reasons = append(gate.Reasons, "coverage_empty")
	}
	gate.Score = clampFloat(score, 0, 1)
	if gate.Score < requiredScore {
		gate.Reasons = append(gate.Reasons, "score_below_threshold")
	}
	gate.Promote = len(gate.Reasons) == 0
	return gate
}

func AssessRecipeRisk(compiled CompiledRecipe, lint RecipeLintReport) RecipeRiskReport {
	report := RecipeRiskReport{RecipeHash: compiled.Hash}
	add := func(score float64, reason string) {
		report.RiskScore += score
		report.Risks = append(report.Risks, reason)
	}
	if !lint.Valid {
		add(0.40, "lint_errors")
	}
	if len(compiled.Steps) > 25 {
		add(0.15, "many_steps")
	}
	for _, step := range compiled.Steps {
		if step.Kind == RecipeStepRegexCapture && looksExpensiveRegex(step.Argument) {
			add(0.25, "expensive_regex")
		}
		if step.Kind == RecipeStepMaxBytes {
			if n, err := strconv.Atoi(step.Argument); err == nil && n > 10*1024*1024 {
				add(0.10, "large_output_limit")
			}
		}
	}
	report.RiskScore = clampFloat(report.RiskScore, 0, 1)
	report.RequiresReview = report.RiskScore >= 0.30
	sort.Strings(report.Risks)
	return report
}

func BenchmarkRecipe(compiled CompiledRecipe, rawSamples map[string]string) RecipeBenchmarkStats {
	keys := make([]string, 0, len(rawSamples))
	for key := range rawSamples {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	stats := RecipeBenchmarkStats{RecipeHash: compiled.Hash, SampleCount: len(keys)}
	latencies := make([]float64, 0, len(keys))
	yields := make([]float64, 0, len(keys))
	tokens := make([]float64, 0, len(keys))
	failures := 0
	rejected := 0
	totalLatency := 0.0
	for _, key := range keys {
		result := ExecuteCompiledRecipe(compiled, rawSamples[key], key)
		latencies = append(latencies, result.ExecutionLatencyMS)
		yields = append(yields, result.YieldRatio)
		tokens = append(tokens, float64(result.TokenCount))
		totalLatency += result.ExecutionLatencyMS
		if result.SignalBytes == 0 {
			failures++
		}
		if result.Rejected {
			rejected++
		}
	}
	sort.Float64s(latencies)
	stats.MeanLatencyMS = meanFloat64(latencies)
	stats.P95LatencyMS = percentileFloat64(latencies, 0.95)
	stats.MeanYield = meanFloat64(yields)
	stats.MeanTokens = meanFloat64(tokens)
	stats.FailureRate = ratioInt(failures, len(keys))
	stats.RejectedRate = ratioInt(rejected, len(keys))
	if totalLatency > 0 {
		stats.ThroughputPerSec = float64(len(keys)) / (totalLatency / 1000.0)
	}
	return stats
}

func BuildRecipeRolloutPlan(gate RecipePromotionGate, risk RecipeRiskReport, benchmark RecipeBenchmarkStats) RecipeRolloutPlan {
	plan := RecipeRolloutPlan{
		RecipeHash:      firstNonEmpty(risk.RecipeHash, benchmark.RecipeHash),
		GeneratedAtUnix: time.Now().Unix(),
		CanaryPercent:   0,
		RequiredSamples: 50,
	}
	plan.Guardrails = []string{"abort_on_empty_rate_gt_0.40", "abort_on_failure_rate_gt_0.35", "abort_on_latency_p95_gt_1000ms"}
	if !gate.Promote {
		plan.Decision = "hold"
		plan.Reasons = append(plan.Reasons, gate.Reasons...)
	}
	if risk.RequiresReview {
		plan.Decision = "manual_review"
		plan.Reasons = append(plan.Reasons, risk.Risks...)
	}
	if benchmark.SampleCount < 10 {
		plan.Decision = "collect_more_samples"
		plan.RequiredSamples = 10 - benchmark.SampleCount
		plan.Reasons = append(plan.Reasons, "insufficient_benchmark_samples")
	}
	if plan.Decision == "" {
		switch {
		case gate.Score >= 0.90 && risk.RiskScore < 0.10:
			plan.Decision = "promote"
			plan.CanaryPercent = 100
		case gate.Score >= 0.75:
			plan.Decision = "canary"
			plan.CanaryPercent = 25
		default:
			plan.Decision = "canary"
			plan.CanaryPercent = 10
		}
	}
	return plan
}

func ClusterRecipeFailures(results []RecipeExecutionResult) []RecipeFailureCluster {
	clusters := make(map[string]*RecipeFailureCluster)
	for _, result := range results {
		reason := result.RejectReason
		if reason == "" && result.SignalBytes == 0 {
			reason = "empty_signal"
		}
		if reason == "" {
			continue
		}
		cluster := clusters[reason]
		if cluster == nil {
			cluster = &RecipeFailureCluster{Reason: reason, RecommendedStep: recommendedStepForFailure(reason)}
			clusters[reason] = cluster
		}
		cluster.Count++
		if len(cluster.URLs) < 10 {
			cluster.URLs = append(cluster.URLs, result.URL)
		}
		if cluster.Representative == "" {
			cluster.Representative = result.URL
		}
	}
	out := make([]RecipeFailureCluster, 0, len(clusters))
	for _, cluster := range clusters {
		out = append(out, *cluster)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Reason < out[j].Reason
	})
	return out
}

func BuildRecipeRepairPlan(report RecipeValidationReport, lint RecipeLintReport, results []RecipeExecutionResult, recipe string) RecipeRepairPlan {
	actions := SuggestRecipePatches(report, lint)
	mutations := MutateRecipe(recipe, report)
	clusters := ClusterRecipeFailures(results)
	plan := RecipeRepairPlan{
		RecipeHash:      report.RecipeHash,
		Actions:         actions,
		Mutations:       mutations,
		FailureClusters: clusters,
		Priority:        report.RecommendedPriority,
	}
	confidence := report.Confidence
	if len(clusters) > 0 {
		confidence = math.Max(confidence, ratioInt(clusters[0].Count, maxInt(1, report.SampleSize)))
	}
	plan.Confidence = clampFloat(confidence, 0, 1)
	if len(actions) == 0 && len(mutations) == 0 && len(clusters) == 0 {
		plan.Priority = 3
	}
	return plan
}

func ScoreRecipeQuality(report RecipeValidationReport, coverage RecipeCoverageReport, trend RecipeTrendSummary, risk RecipeRiskReport) RecipeQualityScorecard {
	card := RecipeQualityScorecard{RecipeHash: report.RecipeHash}
	card.YieldScore = clampFloat(report.MeanYield/MaximumYieldRatio, 0, 1)
	card.CoverageScore = coverage.CoverageRatio
	card.LatencyScore = clampFloat(1.0-report.MedianLatencyMS/2000.0, 0, 1)
	card.StabilityScore = trend.StabilityScore
	card.RiskPenalty = risk.RiskScore
	card.OverallScore = clampFloat(card.YieldScore*0.25+card.CoverageScore*0.25+card.LatencyScore*0.20+card.StabilityScore*0.20-risk.RiskScore*0.30, 0, 1)
	switch {
	case card.OverallScore >= 0.90:
		card.Grade = "A"
	case card.OverallScore >= 0.75:
		card.Grade = "B"
	case card.OverallScore >= 0.60:
		card.Grade = "C"
	case card.OverallScore >= 0.40:
		card.Grade = "D"
	default:
		card.Grade = "F"
	}
	return card
}

func MergeRecipeReports(reports []RecipeValidationReport) RecipeValidationReport {
	merged := RecipeValidationReport{}
	if len(reports) == 0 {
		return merged
	}
	merged.Domain = reports[0].Domain
	merged.TopologyClass = reports[0].TopologyClass
	merged.RecipeHash = reports[0].RecipeHash
	var yieldWeight float64
	var latencyWeight float64
	for _, report := range reports {
		merged.SampleSize += report.SampleSize
		merged.WindowSize += report.WindowSize
		merged.EmptyRate += report.EmptyRate * float64(report.SampleSize)
		merged.FailureRate += report.FailureRate * float64(report.SampleSize)
		merged.TooBroadRate += report.TooBroadRate * float64(report.SampleSize)
		merged.MeanYield += report.MeanYield * float64(report.SampleSize)
		merged.MedianLatencyMS += report.MedianLatencyMS * float64(report.SampleSize)
		yieldWeight += float64(report.SampleSize)
		latencyWeight += float64(report.SampleSize)
		if report.Stale {
			merged.Stale = true
			merged.StaleReason = firstNonEmpty(merged.StaleReason, report.StaleReason)
		}
		if report.Confidence > merged.Confidence {
			merged.Confidence = report.Confidence
		}
		merged.Samples = append(merged.Samples, report.Samples...)
	}
	if yieldWeight > 0 {
		merged.EmptyRate /= yieldWeight
		merged.FailureRate /= yieldWeight
		merged.TooBroadRate /= yieldWeight
		merged.MeanYield /= yieldWeight
	}
	if latencyWeight > 0 {
		merged.MedianLatencyMS /= latencyWeight
	}
	merged.RecommendedAction, merged.RecommendedPriority = recommendedRecipeAction(merged.Stale, merged.StaleReason, merged.Confidence)
	return merged
}

func RankRegistryRecords(records []RecipeRegistryRecord) []RecipeRegistryRecord {
	out := NormalizeRegistryRecords(records)
	sort.Slice(out, func(i, j int) bool {
		left := registryRecordPriority(out[i])
		right := registryRecordPriority(out[j])
		if left != right {
			return left > right
		}
		return recipeRegistryIdentity(out[i]) < recipeRegistryIdentity(out[j])
	})
	return out
}

func PruneStaleRecipes(records []RecipeRegistryRecord, maxStaleAgeSeconds int64, nowUnix int64) []RecipeRegistryRecord {
	if nowUnix <= 0 {
		nowUnix = time.Now().Unix()
	}
	if maxStaleAgeSeconds <= 0 {
		maxStaleAgeSeconds = 30 * 86400
	}
	out := make([]RecipeRegistryRecord, 0, len(records))
	for _, record := range records {
		if record.Stale && record.LastValidatedUnix > 0 && nowUnix-record.LastValidatedUnix > maxStaleAgeSeconds {
			continue
		}
		out = append(out, record)
	}
	return NormalizeRegistryRecords(out)
}

func SelectRecipesForRevalidation(records []RecipeRegistryRecord, limit int, nowUnix int64) []RecipeRevalidationCandidate {
	if limit <= 0 {
		limit = 32
	}
	if nowUnix <= 0 {
		nowUnix = time.Now().Unix()
	}
	candidates := make([]RecipeRevalidationCandidate, 0, len(records))
	for _, record := range records {
		ageDays := 999.0
		if record.LastValidatedUnix > 0 {
			ageDays = float64(nowUnix-record.LastValidatedUnix) / 86400.0
		}
		priority := 0.0
		reason := "age"
		if record.Stale {
			priority += 1.0
			reason = "stale"
		}
		priority += clampFloat(ageDays/30.0, 0, 1) * 0.5
		if record.HistoricalYield < MinimumYieldRatio*5 {
			priority += 0.25
			reason = "low_yield"
		}
		candidates = append(candidates, RecipeRevalidationCandidate{Record: record, Priority: priority, Reason: reason})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Priority != candidates[j].Priority {
			return candidates[i].Priority > candidates[j].Priority
		}
		return recipeRegistryIdentity(candidates[i].Record) < recipeRegistryIdentity(candidates[j].Record)
	})
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates
}

func GenerateRecipeHealthEvents(records []RecipeRegistryRecord, runID string) []RecipeHealthEvent {
	events := make([]RecipeHealthEvent, 0, len(records))
	for _, record := range records {
		events = append(events, RecipeHealthEvent{
			TopologyClass:   record.TopologyClass,
			RecipeHash:      firstNonEmpty(record.RecipeHash, hashRecipe(record.Recipe)),
			SampleCount:     record.SampleCount,
			SuccessCount:    record.SampleCount,
			FailureCount:    0,
			EmptyRate:       0,
			MedianLatencyMS: record.HistoricalLatencyMS,
			Stale:           record.Stale,
			RunID:           runID,
		})
	}
	return events
}

func BuildRecipeManifest(records []RecipeRegistryRecord, nowUnix int64) RecipeManifest {
	if nowUnix <= 0 {
		nowUnix = time.Now().Unix()
	}
	normalized := NormalizeRegistryRecords(records)
	manifest := RecipeManifest{GeneratedAtUnix: nowUnix}
	for _, record := range normalized {
		manifest.Entries = append(manifest.Entries, RecipeManifestEntry{
			Domain:          record.Domain,
			TopologyClass:   record.TopologyClass,
			RecipeHash:      firstNonEmpty(record.RecipeHash, hashRecipe(record.Recipe)),
			Stale:           record.Stale,
			HistoricalYield: record.HistoricalYield,
			SampleCount:     record.SampleCount,
		})
	}
	payload, _ := json.Marshal(manifest.Entries)
	manifest.ManifestHash = hashRecipe(string(payload))
	return manifest
}

func CheckRecipeCompatibility(record RecipeRegistryRecord, supportedTopologies map[string]bool) RecipeCompatibilityReport {
	report := RecipeCompatibilityReport{Compatible: true}
	add := func(code, severity, message string) {
		report.Issues = append(report.Issues, RecipeCompatibilityIssue{RecipeHash: firstNonEmpty(record.RecipeHash, hashRecipe(record.Recipe)), Code: code, Severity: severity, Message: message})
		if severity == "error" {
			report.ErrorCount++
			report.Compatible = false
		} else {
			report.WarnCount++
		}
	}
	if normalizeDomain(record.Domain) == "" {
		add("empty_domain", "error", "record domain is empty")
	}
	if record.TopologyClass == "" {
		add("empty_topology", "error", "record topology class is empty")
	} else if supportedTopologies != nil && !supportedTopologies[record.TopologyClass] {
		add("unsupported_topology", "error", "topology is not supported by this runtime")
	}
	_, lint := CompileRecipe(record.TopologyClass, record.Recipe, 0)
	for _, issue := range lint.Issues {
		if issue.Severity == "error" {
			add("lint_"+issue.Code, "error", issue.Message)
		} else {
			add("lint_"+issue.Code, "warning", issue.Message)
		}
	}
	if record.RecipeHash != "" && record.Recipe != "" && record.RecipeHash != hashRecipe(record.Recipe) {
		add("hash_mismatch", "error", "stored recipe hash does not match recipe body")
	}
	if record.SampleCount == 0 {
		add("no_samples", "warning", "record has no validation samples")
	}
	return report
}

func BuildRecipeMigrationPlan(oldRecords []RecipeRegistryRecord, newRecords []RecipeRegistryRecord) RecipeMigrationPlan {
	diff := DiffRecipeRegistries(oldRecords, newRecords)
	plan := RecipeMigrationPlan{RequiresBackup: diff.RequiresReload, GeneratedAtUnix: time.Now().Unix()}
	oldMap := make(map[string]RecipeRegistryRecord)
	newMap := make(map[string]RecipeRegistryRecord)
	for _, record := range oldRecords {
		oldMap[recipeRegistryIdentity(record)] = record
	}
	for _, record := range newRecords {
		newMap[recipeRegistryIdentity(record)] = record
	}
	for _, key := range diff.Removed {
		record := oldMap[key]
		plan.Steps = append(plan.Steps, RecipeMigrationStep{Action: "remove", Domain: record.Domain, TopologyClass: record.TopologyClass, RecipeHash: record.RecipeHash, Reason: "record_removed"})
	}
	for _, key := range diff.Added {
		record := newMap[key]
		plan.Steps = append(plan.Steps, RecipeMigrationStep{Action: "add", Domain: record.Domain, TopologyClass: record.TopologyClass, RecipeHash: firstNonEmpty(record.RecipeHash, hashRecipe(record.Recipe)), Reason: "record_added"})
	}
	for _, key := range diff.Changed {
		record := newMap[key]
		plan.Steps = append(plan.Steps, RecipeMigrationStep{Action: "replace", Domain: record.Domain, TopologyClass: record.TopologyClass, RecipeHash: firstNonEmpty(record.RecipeHash, hashRecipe(record.Recipe)), Reason: "record_changed"})
	}
	sort.Slice(plan.Steps, func(i, j int) bool {
		if plan.Steps[i].Action != plan.Steps[j].Action {
			return plan.Steps[i].Action < plan.Steps[j].Action
		}
		return plan.Steps[i].Domain < plan.Steps[j].Domain
	})
	payload, _ := json.Marshal(plan.Steps)
	plan.PlanHash = hashRecipe(string(payload))
	return plan
}

func DiffRecipeManifests(oldManifest RecipeManifest, newManifest RecipeManifest) RecipeManifestDiff {
	oldMap := make(map[string]RecipeManifestEntry, len(oldManifest.Entries))
	newMap := make(map[string]RecipeManifestEntry, len(newManifest.Entries))
	for _, entry := range oldManifest.Entries {
		oldMap[manifestEntryKey(entry)] = entry
	}
	for _, entry := range newManifest.Entries {
		newMap[manifestEntryKey(entry)] = entry
	}
	diff := RecipeManifestDiff{}
	for key, entry := range newMap {
		old, ok := oldMap[key]
		if !ok {
			diff.Added = append(diff.Added, entry)
			continue
		}
		if old.RecipeHash != entry.RecipeHash || old.Stale != entry.Stale || math.Abs(old.HistoricalYield-entry.HistoricalYield) > 0.001 {
			diff.Changed = append(diff.Changed, entry)
		} else {
			diff.Unchanged++
		}
	}
	for key, entry := range oldMap {
		if _, ok := newMap[key]; !ok {
			diff.Removed = append(diff.Removed, entry)
		}
	}
	sortManifestEntries(diff.Added)
	sortManifestEntries(diff.Removed)
	sortManifestEntries(diff.Changed)
	diff.RequiresRestart = len(diff.Added) > 0 || len(diff.Removed) > 0 || len(diff.Changed) > 0
	return diff
}

func ApplyRolloutDecision(record RecipeRegistryRecord, rollout RecipeRolloutPlan, nowUnix int64) RecipeRegistryRecord {
	if nowUnix <= 0 {
		nowUnix = time.Now().Unix()
	}
	out := record
	switch rollout.Decision {
	case "promote":
		out.Stale = false
		out.LastValidatedUnix = nowUnix
	case "canary":
		out.LastValidatedUnix = nowUnix
	case "hold", "manual_review", "collect_more_samples":
		out.Stale = true
	}
	if out.RecipeHash == "" && out.Recipe != "" {
		out.RecipeHash = hashRecipe(out.Recipe)
	}
	return out
}

func SynthesizeValidationSamples(results []RecipeExecutionResult, capturedAtUnix int64) []RecipeYieldSample {
	if capturedAtUnix <= 0 {
		capturedAtUnix = time.Now().Unix()
	}
	out := make([]RecipeYieldSample, 0, len(results))
	for _, result := range results {
		out = append(out, RecipeYieldSample{
			URL:            result.URL,
			TopologyClass:  result.TopologyClass,
			RawBytes:       result.RawBytes,
			SignalBytes:    result.SignalBytes,
			YieldRatio:     result.YieldRatio,
			LatencyMS:      result.ExecutionLatencyMS,
			Succeeded:      !result.Rejected && result.SignalBytes > 0,
			Empty:          result.SignalBytes == 0,
			CapturedAtUnix: capturedAtUnix,
		})
	}
	return out
}

func BuildRecipeAuditTrail(records []RecipeRegistryRecord, action string, actor string, reason string, nowUnix int64) RecipeAuditTrail {
	if nowUnix <= 0 {
		nowUnix = time.Now().Unix()
	}
	if actor == "" {
		actor = "preparser.recipe_validator"
	}
	trail := RecipeAuditTrail{}
	for _, record := range NormalizeRegistryRecords(records) {
		trail.Entries = append(trail.Entries, RecipeAuditEntry{
			UnixTime:      nowUnix,
			RecipeHash:    firstNonEmpty(record.RecipeHash, hashRecipe(record.Recipe)),
			Action:        action,
			Actor:         actor,
			Reason:        reason,
			TopologyClass: record.TopologyClass,
		})
	}
	payload, _ := json.Marshal(trail.Entries)
	trail.Digest = hashRecipe(string(payload))
	return trail
}

func CheckRecipeInvariants(records []RecipeRegistryRecord) RecipeInvariantSummary {
	summary := RecipeInvariantSummary{RecordCount: len(records), Healthy: true}
	for _, record := range records {
		if strings.TrimSpace(record.Domain) == "" {
			summary.MissingDomains++
			summary.Warnings = append(summary.Warnings, "missing domain")
		}
		if strings.TrimSpace(record.TopologyClass) == "" {
			summary.MissingTopologies++
			summary.Warnings = append(summary.Warnings, "missing topology")
		}
		if strings.TrimSpace(record.Recipe) == "" {
			summary.EmptyRecipes++
			summary.Warnings = append(summary.Warnings, "empty recipe")
		}
		if record.RecipeHash != "" && record.Recipe != "" && record.RecipeHash != hashRecipe(record.Recipe) {
			summary.HashMismatches++
			summary.Warnings = append(summary.Warnings, "hash mismatch for "+record.Domain+"/"+record.TopologyClass)
		}
	}
	if summary.HashMismatches > 0 || summary.EmptyRecipes > 0 || summary.MissingDomains > 0 || summary.MissingTopologies > 0 {
		summary.Healthy = false
	}
	sort.Strings(summary.Warnings)
	return summary
}

func RecipeRecordsEqual(a RecipeRegistryRecord, b RecipeRegistryRecord) bool {
	return normalizeDomain(a.Domain) == normalizeDomain(b.Domain) &&
		strings.TrimSpace(a.TopologyClass) == strings.TrimSpace(b.TopologyClass) &&
		firstNonEmpty(a.RecipeHash, hashRecipe(a.Recipe)) == firstNonEmpty(b.RecipeHash, hashRecipe(b.Recipe)) &&
		math.Abs(a.HistoricalYield-b.HistoricalYield) < 0.000001 &&
		a.Stale == b.Stale
}

func RecipeValidatorCapabilities() []string {
	return []string{
		"dsl_compile",
		"window_validation",
		"cross_validation",
		"registry_diff",
		"rollout_gate",
		"risk_assessment",
		"audit_trail",
	}
}

func (e RecipeHealthEvent) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "recipe_health", Component: "preparser.recipe_validator", Payload: e}
}

func (e RecipeStaleEvent) BridgeEvent() BridgeRequest {
	return BridgeRequest{Topic: "recipe_stale", Component: "preparser.recipe_validator", Payload: e}
}

func hashRecipe(recipe string) string {
	sum := sha256.Sum256([]byte(recipe))
	return hex.EncodeToString(sum[:])
}

func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := append([]float64(nil), vals...)
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && cp[j] < cp[j-1]; j-- {
			cp[j], cp[j-1] = cp[j-1], cp[j]
		}
	}
	mid := len(cp) / 2
	if len(cp)%2 == 1 {
		return cp[mid]
	}
	return (cp[mid-1] + cp[mid]) / 2
}

func staleConfidence(h RecipeHealthEvent) float64 {
	if h.SampleCount == 0 {
		return 0.6
	}
	failRate := float64(h.FailureCount) / float64(h.SampleCount)
	conf := h.EmptyRate
	if failRate > conf {
		conf = failRate
	}
	if conf < 0.6 {
		return 0.6
	}
	if conf > 1 {
		return 1
	}
	return conf
}

func normalizeValidationOptions(record RecipeRegistryRecord, opts RecipeValidationOptions) RecipeValidationOptions {
	if opts.Domain == "" {
		opts.Domain = record.Domain
	}
	if opts.TopologyClass == "" {
		opts.TopologyClass = record.TopologyClass
	}
	if opts.HistoricalYield <= 0 {
		opts.HistoricalYield = record.HistoricalYield
	}
	if opts.MinimumYieldRatio <= 0 {
		opts.MinimumYieldRatio = MinimumYieldRatio
	}
	if opts.MaximumYieldRatio <= 0 {
		opts.MaximumYieldRatio = MaximumYieldRatio
	}
	if opts.StaleThresholdFactor <= 0 {
		opts.StaleThresholdFactor = StaleThresholdFactor
	}
	if opts.WindowSize <= 0 {
		opts.WindowSize = StaleWindowSize
	}
	if opts.NowUnix <= 0 {
		opts.NowUnix = time.Now().Unix()
	}
	return opts
}

func selectValidationWindow(samples []RecipeYieldSample, windowSize int) []RecipeYieldSample {
	if windowSize <= 0 {
		windowSize = StaleWindowSize
	}
	cp := append([]RecipeYieldSample(nil), samples...)
	sort.SliceStable(cp, func(i, j int) bool {
		if cp[i].CapturedAtUnix != cp[j].CapturedAtUnix {
			return cp[i].CapturedAtUnix > cp[j].CapturedAtUnix
		}
		return cp[i].URL < cp[j].URL
	})
	if len(cp) > windowSize {
		cp = cp[:windowSize]
	}
	sort.SliceStable(cp, func(i, j int) bool {
		if cp[i].CapturedAtUnix != cp[j].CapturedAtUnix {
			return cp[i].CapturedAtUnix < cp[j].CapturedAtUnix
		}
		return cp[i].URL < cp[j].URL
	})
	return cp
}

func normalizeYieldSample(sample RecipeYieldSample) RecipeYieldSample {
	if sample.RawBytes < 0 {
		sample.RawBytes = 0
	}
	if sample.SignalBytes < 0 {
		sample.SignalBytes = 0
	}
	if sample.YieldRatio <= 0 && sample.RawBytes > 0 {
		sample.YieldRatio = float64(sample.SignalBytes) / float64(sample.RawBytes)
	}
	if sample.SignalBytes == 0 {
		sample.Empty = true
	}
	if sample.SignalBytes > 0 && sample.RawBytes > 0 && !sample.Empty {
		sample.Succeeded = true
	}
	return sample
}

func recipeStalenessDecision(mean float64, median float64, emptyRate float64, failureRate float64, tooBroadRate float64, opts RecipeValidationOptions) (bool, string) {
	if lenReason := windowSizeReason(opts.WindowSize); lenReason != "" {
		return true, lenReason
	}
	if failureRate > 0.5 {
		return true, "failure_rate_high"
	}
	if emptyRate >= 0.4 {
		return true, "empty_rate_high"
	}
	if tooBroadRate >= 0.25 {
		return true, "yield_too_broad"
	}
	if median < opts.MinimumYieldRatio && mean < opts.MinimumYieldRatio*2 {
		return true, "yield_below_minimum"
	}
	if opts.HistoricalYield > 0 && mean < opts.HistoricalYield*opts.StaleThresholdFactor {
		return true, "yield_dropped"
	}
	if mean > opts.MaximumYieldRatio {
		return true, "recipe_too_broad"
	}
	return false, "healthy"
}

func windowSizeReason(windowSize int) string {
	if windowSize <= 0 {
		return "window_missing"
	}
	return ""
}

func recipeValidationConfidence(sampleSize int, mean float64, median float64, emptyRate float64, failureRate float64, tooBroadRate float64, opts RecipeValidationOptions) float64 {
	sampleScore := math.Min(1, float64(sampleSize)/float64(maxInt(opts.WindowSize, 1)))
	yieldScore := 0.5
	if opts.HistoricalYield > 0 {
		yieldScore = clampFloat(math.Abs(mean-opts.HistoricalYield)/opts.HistoricalYield, 0, 1)
	} else if mean > 0 {
		yieldScore = clampFloat(mean/opts.MaximumYieldRatio, 0, 1)
	}
	qualitySignal := math.Max(emptyRate, math.Max(failureRate, tooBroadRate))
	if median < opts.MinimumYieldRatio {
		qualitySignal = math.Max(qualitySignal, 0.7)
	}
	return clampFloat(0.35*sampleScore+0.35*qualitySignal+0.30*yieldScore, 0.05, 1)
}

func recommendedRecipeAction(stale bool, reason string, confidence float64) (string, int) {
	if !stale {
		return "keep_recipe", 2
	}
	switch reason {
	case "failure_rate_high", "empty_rate_high", "yield_below_minimum":
		if confidence >= 0.75 {
			return "recompile_recipe_now", 0
		}
		return "queue_recipe_recompile", 1
	case "yield_too_broad", "recipe_too_broad":
		return "tighten_recipe_selectors", 1
	case "yield_dropped":
		return "refresh_recipe_window", 1
	default:
		return "inspect_recipe", 1
	}
}

func splitRecipeStatements(recipe string) []string {
	recipe = strings.ReplaceAll(recipe, "\r\n", "\n")
	var out []string
	for _, line := range strings.Split(recipe, "\n") {
		for _, part := range strings.Split(line, ";") {
			part = strings.TrimSpace(part)
			if part != "" {
				out = append(out, part)
			}
		}
	}
	return out
}

func parseRecipeStatement(stmt string, line int) (RecipeStep, error) {
	head, tail, ok := strings.Cut(stmt, ":")
	if !ok {
		head = stmt
		tail = ""
	}
	kind := RecipeStepKind(strings.TrimSpace(strings.ToLower(head)))
	step := RecipeStep{Kind: kind, Line: line, Raw: stmt, Confidence: 1.0}
	switch kind {
	case RecipeStepStripHTML, RecipeStepCollapseWS, RecipeStepFirstLine, RecipeStepLastLine:
		return step, nil
	case RecipeStepKeepBetween, RecipeStepReplace:
		arg, value, ok := splitRecipePair(tail)
		if !ok {
			return step, errors.New(string(kind) + " requires 'left => right'")
		}
		step.Argument = arg
		step.Value = value
		return step, nil
	case RecipeStepSelect, RecipeStepDrop, RecipeStepMinTokens, RecipeStepMaxBytes, RecipeStepRequire, RecipeStepReject, RecipeStepRegexCapture:
		step.Argument = strings.TrimSpace(tail)
		return step, nil
	default:
		step.Argument = strings.TrimSpace(tail)
		return step, nil
	}
}

func splitRecipePair(raw string) (string, string, bool) {
	for _, sep := range []string{"=>", "->", ","} {
		left, right, ok := strings.Cut(raw, sep)
		if ok {
			left = strings.TrimSpace(left)
			right = strings.TrimSpace(right)
			return left, right, left != "" && right != ""
		}
	}
	return "", "", false
}

func recipeStepKindKnown(kind RecipeStepKind) bool {
	switch kind {
	case RecipeStepSelect, RecipeStepDrop, RecipeStepKeepBetween, RecipeStepStripHTML, RecipeStepCollapseWS, RecipeStepReplace, RecipeStepMinTokens, RecipeStepMaxBytes, RecipeStepRequire, RecipeStepReject, RecipeStepFirstLine, RecipeStepLastLine, RecipeStepRegexCapture:
		return true
	default:
		return false
	}
}

func renderRecipeSteps(steps []RecipeStep) string {
	lines := make([]string, 0, len(steps))
	for _, step := range steps {
		switch step.Kind {
		case RecipeStepKeepBetween, RecipeStepReplace:
			lines = append(lines, string(step.Kind)+":"+step.Argument+"=>"+step.Value)
		case RecipeStepStripHTML, RecipeStepCollapseWS, RecipeStepFirstLine, RecipeStepLastLine:
			lines = append(lines, string(step.Kind))
		default:
			lines = append(lines, string(step.Kind)+":"+step.Argument)
		}
	}
	return strings.Join(lines, "\n")
}

func executeRecipeStep(step RecipeStep, input string) (string, bool, string) {
	switch step.Kind {
	case RecipeStepSelect:
		return selectContainingLines(input, step.Argument), false, ""
	case RecipeStepDrop:
		return dropContainingLines(input, step.Argument), false, ""
	case RecipeStepKeepBetween:
		return keepBetweenText(input, step.Argument, step.Value), false, ""
	case RecipeStepStripHTML:
		return stripSimpleHTML(input), false, ""
	case RecipeStepCollapseWS:
		return collapseRecipeWhitespace(input), false, ""
	case RecipeStepReplace:
		return strings.ReplaceAll(input, step.Argument, step.Value), false, ""
	case RecipeStepMinTokens:
		n, _ := strconv.Atoi(strings.TrimSpace(step.Argument))
		if countTokens(input) < n {
			return "", true, "min_tokens_not_met"
		}
		return input, false, ""
	case RecipeStepMaxBytes:
		n, _ := strconv.Atoi(strings.TrimSpace(step.Argument))
		if n > 0 && len([]byte(input)) > n {
			return string([]byte(input)[:n]), false, ""
		}
		return input, false, ""
	case RecipeStepRequire:
		if !strings.Contains(strings.ToLower(input), strings.ToLower(step.Argument)) {
			return "", true, "required_marker_missing"
		}
		return input, false, ""
	case RecipeStepReject:
		if strings.Contains(strings.ToLower(input), strings.ToLower(step.Argument)) {
			return "", true, "reject_marker_present"
		}
		return input, false, ""
	case RecipeStepFirstLine:
		lines := nonEmptyRecipeLines(input)
		if len(lines) == 0 {
			return "", false, ""
		}
		return lines[0], false, ""
	case RecipeStepLastLine:
		lines := nonEmptyRecipeLines(input)
		if len(lines) == 0 {
			return "", false, ""
		}
		return lines[len(lines)-1], false, ""
	case RecipeStepRegexCapture:
		return regexCapture(input, step.Argument), false, ""
	default:
		return input, false, ""
	}
}

func selectContainingLines(input string, needle string) string {
	needle = strings.ToLower(strings.TrimSpace(needle))
	if needle == "" {
		return input
	}
	var out []string
	for _, line := range strings.Split(input, "\n") {
		if strings.Contains(strings.ToLower(line), needle) {
			out = append(out, line)
		}
	}
	return strings.Join(out, "\n")
}

func dropContainingLines(input string, needle string) string {
	needle = strings.ToLower(strings.TrimSpace(needle))
	if needle == "" {
		return input
	}
	var out []string
	for _, line := range strings.Split(input, "\n") {
		if !strings.Contains(strings.ToLower(line), needle) {
			out = append(out, line)
		}
	}
	return strings.Join(out, "\n")
}

func keepBetweenText(input string, start string, end string) string {
	if start == "" || end == "" {
		return input
	}
	lower := strings.ToLower(input)
	startLower := strings.ToLower(start)
	endLower := strings.ToLower(end)
	i := strings.Index(lower, startLower)
	if i < 0 {
		return ""
	}
	i += len(start)
	j := strings.Index(lower[i:], endLower)
	if j < 0 {
		return strings.TrimSpace(input[i:])
	}
	return strings.TrimSpace(input[i : i+j])
}

func stripSimpleHTML(input string) string {
	var out strings.Builder
	inTag := false
	for _, r := range input {
		switch r {
		case '<':
			inTag = true
			out.WriteRune(' ')
		case '>':
			inTag = false
		default:
			if !inTag {
				out.WriteRune(r)
			}
		}
	}
	return out.String()
}

func collapseRecipeWhitespace(input string) string {
	return strings.Join(strings.Fields(input), " ")
}

func nonEmptyRecipeLines(input string) []string {
	var out []string
	for _, line := range strings.Split(input, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

func regexCapture(input string, pattern string) string {
	if strings.TrimSpace(pattern) == "" {
		return input
	}
	rx, err := regexp.Compile(pattern)
	if err != nil {
		return ""
	}
	matches := rx.FindAllStringSubmatch(input, -1)
	if len(matches) == 0 {
		return ""
	}
	var out []string
	for _, match := range matches {
		if len(match) > 1 {
			out = append(out, strings.Join(match[1:], " "))
		} else if len(match) == 1 {
			out = append(out, match[0])
		}
	}
	return strings.Join(out, "\n")
}

func recipeRegistryKey(record RecipeRegistryRecord) string {
	key := record.Domain + "\x00" + record.TopologyClass + "\x00" + record.RecipeHash
	if record.RecipeHash == "" {
		key = record.Domain + "\x00" + record.TopologyClass + "\x00" + hashRecipe(record.Recipe)
	}
	return key
}

func buildTrendPoint(start int64, end int64, samples []RecipeYieldSample) RecipeTrendPoint {
	yields := make([]float64, 0, len(samples))
	failures := 0
	empty := 0
	for _, sample := range samples {
		sample = normalizeYieldSample(sample)
		yields = append(yields, sample.YieldRatio)
		if !sample.Succeeded {
			failures++
		}
		if sample.Empty || sample.SignalBytes == 0 {
			empty++
		}
	}
	sort.Float64s(yields)
	return RecipeTrendPoint{
		WindowStartUnix: start,
		WindowEndUnix:   end,
		MeanYield:       meanFloat64(yields),
		MedianYield:     percentileFloat64(yields, 0.50),
		FailureRate:     ratioInt(failures, len(samples)),
		EmptyRate:       ratioInt(empty, len(samples)),
		SampleCount:     len(samples),
	}
}

func trendSlope(points []RecipeTrendPoint, pick func(RecipeTrendPoint) float64) float64 {
	if len(points) < 2 {
		return 0
	}
	n := float64(len(points))
	var sumX, sumY, sumXY, sumXX float64
	for i, point := range points {
		x := float64(i)
		y := pick(point)
		sumX += x
		sumY += y
		sumXY += x * y
		sumXX += x * x
	}
	denom := n*sumXX - sumX*sumX
	if denom == 0 {
		return 0
	}
	return (n*sumXY - sumX*sumY) / denom
}

func recipeTrendStability(points []RecipeTrendPoint) float64 {
	if len(points) == 0 {
		return 0
	}
	mean := 0.0
	for _, point := range points {
		mean += point.MeanYield
	}
	mean /= float64(len(points))
	var variance float64
	for _, point := range points {
		d := point.MeanYield - mean
		variance += d * d
	}
	variance /= float64(len(points))
	return clampFloat(1.0-math.Sqrt(variance)*10.0, 0, 1)
}

func candidateSelectorsForZone(zone SignalZone) []string {
	selectors := make([]string, 0, 4)
	switch zone.Type {
	case ZoneCode:
		selectors = append(selectors, "pre code", "code")
	case ZoneTable:
		selectors = append(selectors, "table", "[role=table]")
	case ZoneHeading:
		selectors = append(selectors, "h1", "h2", "h3")
	case ZoneList:
		selectors = append(selectors, "ul li", "ol li")
	case ZoneQuote:
		selectors = append(selectors, "blockquote")
	default:
		selectors = append(selectors, "main", "article", "[data-content]")
	}
	if zone.TopologyClass == TopologySaaSDocs {
		selectors = append(selectors, "main article", ".docs-content")
	}
	if zone.TopologyClass == TopologyNewsArticle {
		selectors = append(selectors, "article", ".article-body")
	}
	return uniqueStrings(selectors)
}

func recipeABScore(report RecipeValidationReport) float64 {
	if report.SampleSize == 0 {
		return 0
	}
	yieldScore := clampFloat(report.MeanYield/MaximumYieldRatio, 0, 1)
	return clampFloat(yieldScore*0.45+(1-report.FailureRate)*0.25+(1-report.EmptyRate)*0.20+(1-report.TooBroadRate)*0.10, 0, 1)
}

func crossValidationYieldVariance(reports []RecipeValidationReport) float64 {
	if len(reports) == 0 {
		return 0
	}
	mean := 0.0
	for _, report := range reports {
		mean += report.MeanYield
	}
	mean /= float64(len(reports))
	var variance float64
	for _, report := range reports {
		d := report.MeanYield - mean
		variance += d * d
	}
	return variance / float64(len(reports))
}

func recipeRegistryIdentity(record RecipeRegistryRecord) string {
	return normalizeDomain(record.Domain) + "\x00" + strings.TrimSpace(record.TopologyClass)
}

func looksExpensiveRegex(pattern string) bool {
	if strings.Count(pattern, ".*") >= 2 {
		return true
	}
	if strings.Contains(pattern, "(.+") || strings.Contains(pattern, "(.*") {
		return true
	}
	return len(pattern) > 500
}

func uniqueStrings(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, item := range in {
		item = strings.TrimSpace(item)
		if item == "" || seen[item] {
			continue
		}
		seen[item] = true
		out = append(out, item)
	}
	sort.Strings(out)
	return out
}

func recommendedStepForFailure(reason string) string {
	switch reason {
	case "empty_signal":
		return "select:main; select:article"
	case "min_tokens_not_met":
		return "min_tokens:3"
	case "required_marker_missing":
		return "require:<"
	case "reject_marker_present":
		return "drop:nav; drop:footer"
	default:
		return "inspect_recipe"
	}
}

func registryRecordPriority(record RecipeRegistryRecord) float64 {
	priority := 0.0
	if record.Stale {
		priority += 1.0
	}
	priority += clampFloat(1.0-record.HistoricalYield/MaximumYieldRatio, 0, 1) * 0.4
	if record.SampleCount < StaleWindowSize {
		priority += 0.2
	}
	return priority
}

func manifestEntryKey(entry RecipeManifestEntry) string {
	return normalizeDomain(entry.Domain) + "\x00" + strings.TrimSpace(entry.TopologyClass)
}

func sortManifestEntries(entries []RecipeManifestEntry) {
	sort.Slice(entries, func(i, j int) bool {
		return manifestEntryKey(entries[i]) < manifestEntryKey(entries[j])
	})
}

func meanFloat64(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	total := 0.0
	for _, v := range vals {
		total += v
	}
	return total / float64(len(vals))
}

func percentileFloat64(sortedVals []float64, p float64) float64 {
	if len(sortedVals) == 0 {
		return 0
	}
	if p <= 0 {
		return sortedVals[0]
	}
	if p >= 1 {
		return sortedVals[len(sortedVals)-1]
	}
	pos := p * float64(len(sortedVals)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sortedVals[lo]
	}
	weight := pos - float64(lo)
	return sortedVals[lo]*(1-weight) + sortedVals[hi]*weight
}

func ratioInt(n int, d int) float64 {
	if d <= 0 {
		return 0
	}
	return float64(n) / float64(d)
}

func updateHistoricalYield(previous float64, current float64, previousCount int, currentCount int) float64 {
	if currentCount <= 0 {
		return previous
	}
	if previousCount <= 0 {
		return current
	}
	totalCount := previousCount + currentCount
	return (previous*float64(previousCount) + current*float64(currentCount)) / float64(totalCount)
}

func firstNonEmpty(a string, b string) string {
	if a != "" {
		return a
	}
	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
