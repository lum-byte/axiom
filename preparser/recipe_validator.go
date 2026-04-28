package preparser

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"sort"
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
	Domain             string  `json:"domain"`
	TopologyClass      string  `json:"topology_class"`
	Recipe             string  `json:"recipe"`
	RecipeHash         string  `json:"recipe_hash"`
	HistoricalYield    float64 `json:"historical_yield"`
	HistoricalLatencyMS float64 `json:"historical_latency_ms"`
	LastValidatedUnix  int64   `json:"last_validated_unix"`
	SampleCount        int     `json:"sample_count"`
	Stale              bool    `json:"stale"`
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
