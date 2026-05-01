package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultWorkers            = 10
	defaultMaxBodyBytes       = 2 * 1024 * 1024
	defaultRequestTimeoutMS   = 15000
	defaultResultLimit        = 12
	defaultUserAgent          = "AxiomCrawlDaemon/1.0.5 (+https://local.axiom.invalid)"
	defaultPageRankDamping    = 0.85
	defaultPageRankIterations = 8
	maxOutlinksPerPage        = 96
	maxCandidatesPerQuery     = 512
)

var (
	titleRE = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	tagRE   = regexp.MustCompile(`(?s)<[^>]+>`)
	spaceRE = regexp.MustCompile(`\s+`)
	hrefRE  = regexp.MustCompile(`(?is)<a\s+[^>]*href=["']([^"']+)["']`)
	wordRE  = regexp.MustCompile(`[a-z0-9][a-z0-9'-]*`)
)

type Config struct {
	Workers            int
	MaxBodyBytes       int64
	RequestTimeoutMS   int
	UserAgent          string
	PageRankDamping    float64
	PageRankIterations int
}

type Request struct {
	ID           string      `json:"id,omitempty"`
	Op           string      `json:"op"`
	Query        string      `json:"query,omitempty"`
	Candidates   []Candidate `json:"candidates,omitempty"`
	Limit        int         `json:"limit,omitempty"`
	TimeoutMS    int         `json:"timeout_ms,omitempty"`
	MaxBodyBytes int64       `json:"max_body_bytes,omitempty"`
}

type Response struct {
	ID        string         `json:"id,omitempty"`
	Status    string         `json:"status"`
	Message   string         `json:"message,omitempty"`
	Data      map[string]any `json:"data,omitempty"`
	ErrorType string         `json:"error_type,omitempty"`
}

type Candidate struct {
	URL    string  `json:"url"`
	Domain string  `json:"domain,omitempty"`
	Title  string  `json:"title,omitempty"`
	Reason string  `json:"reason,omitempty"`
	Score  float64 `json:"score,omitempty"`
}

type FetchResult struct {
	URL          string   `json:"url"`
	Domain       string   `json:"domain"`
	Title        string   `json:"title"`
	StatusCode   int      `json:"status_code"`
	FetchMode    string   `json:"fetch_mode"`
	Bytes        int      `json:"bytes"`
	Snippet      string   `json:"snippet"`
	Error        string   `json:"error,omitempty"`
	Outlinks     []string `json:"outlinks,omitempty"`
	PageRank     float64  `json:"page_rank"`
	Score        float64  `json:"score"`
	WorkerID     int      `json:"worker_id"`
	FetchedUnix  int64    `json:"fetched_unix"`
	DurationMS   int64    `json:"duration_ms"`
	CandidateKey string   `json:"candidate_key"`
}

type job struct {
	ctx       context.Context
	query     string
	candidate Candidate
	maxBytes  int64
	resultCh  chan<- FetchResult
}

type Daemon struct {
	cfg            Config
	client         *http.Client
	jobs           chan job
	ranker         *PageRanker
	started        time.Time
	workersStarted atomic.Int64
	queriesHandled atomic.Int64
	jobsHandled    atomic.Int64
	jobsFailed     atomic.Int64
	shutdownOnce   sync.Once
}

func NewDaemon(cfg Config) *Daemon {
	cfg = normalizeConfig(cfg)
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConns:          cfg.Workers * 4,
		MaxIdleConnsPerHost:   max(2, cfg.Workers),
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   8 * time.Second,
		ResponseHeaderTimeout: time.Duration(cfg.RequestTimeoutMS) * time.Millisecond,
		DialContext: (&net.Dialer{
			Timeout:   6 * time.Second,
			KeepAlive: 60 * time.Second,
		}).DialContext,
	}
	d := &Daemon{
		cfg:     cfg,
		client:  &http.Client{Transport: transport},
		jobs:    make(chan job, cfg.Workers*8),
		ranker:  NewPageRanker(cfg.PageRankDamping, cfg.PageRankIterations),
		started: time.Now(),
	}
	for workerID := 1; workerID <= cfg.Workers; workerID++ {
		d.workersStarted.Add(1)
		go d.worker(workerID)
	}
	return d
}

func normalizeConfig(cfg Config) Config {
	if cfg.Workers <= 0 {
		cfg.Workers = defaultWorkers
	}
	if cfg.Workers > 512 {
		cfg.Workers = 512
	}
	if cfg.MaxBodyBytes <= 0 {
		cfg.MaxBodyBytes = defaultMaxBodyBytes
	}
	if cfg.RequestTimeoutMS <= 0 {
		cfg.RequestTimeoutMS = defaultRequestTimeoutMS
	}
	if strings.TrimSpace(cfg.UserAgent) == "" {
		cfg.UserAgent = defaultUserAgent
	}
	if cfg.PageRankDamping <= 0 || cfg.PageRankDamping >= 1 {
		cfg.PageRankDamping = defaultPageRankDamping
	}
	if cfg.PageRankIterations <= 0 {
		cfg.PageRankIterations = defaultPageRankIterations
	}
	return cfg
}

func (d *Daemon) Shutdown() {
	d.shutdownOnce.Do(func() {
		close(d.jobs)
		if transport, ok := d.client.Transport.(*http.Transport); ok {
			transport.CloseIdleConnections()
		}
	})
}

func (d *Daemon) worker(workerID int) {
	for work := range d.jobs {
		start := time.Now()
		result := d.fetch(work.ctx, workerID, work.query, work.candidate, work.maxBytes)
		result.DurationMS = time.Since(start).Milliseconds()
		if result.Error != "" {
			d.jobsFailed.Add(1)
		}
		d.jobsHandled.Add(1)
		select {
		case work.resultCh <- result:
		case <-work.ctx.Done():
		}
	}
}

func (d *Daemon) Handle(ctx context.Context, req Request) Response {
	switch strings.ToLower(strings.TrimSpace(req.Op)) {
	case "status", "":
		return d.status(req.ID)
	case "query", "crawl", "search":
		results, telemetry, err := d.Query(ctx, req)
		if err != nil {
			return Response{ID: req.ID, Status: "error", Message: err.Error(), ErrorType: "QueryError"}
		}
		data := map[string]any{
			"results":   results,
			"telemetry": telemetry,
			"pagerank":  d.ranker.Telemetry(),
		}
		return Response{ID: req.ID, Status: "ok", Message: "query complete", Data: data}
	case "shutdown", "quit":
		go d.Shutdown()
		return Response{ID: req.ID, Status: "ok", Message: "shutdown accepted", Data: map[string]any{"uptime_ms": time.Since(d.started).Milliseconds()}}
	default:
		return Response{ID: req.ID, Status: "error", Message: "unknown op: " + req.Op, ErrorType: "UnknownOperation"}
	}
}

func (d *Daemon) status(id string) Response {
	return Response{
		ID:      id,
		Status:  "ok",
		Message: "resident crawl daemon ready",
		Data: map[string]any{
			"workers":         d.cfg.Workers,
			"workers_started": d.workersStarted.Load(),
			"queries_handled": d.queriesHandled.Load(),
			"jobs_handled":    d.jobsHandled.Load(),
			"jobs_failed":     d.jobsFailed.Load(),
			"uptime_ms":       time.Since(d.started).Milliseconds(),
			"queue_depth":     len(d.jobs),
			"pagerank":        d.ranker.Telemetry(),
		},
	}
}

func (d *Daemon) Query(parent context.Context, req Request) ([]FetchResult, map[string]any, error) {
	candidates := normalizeCandidates(req.Candidates)
	if len(candidates) == 0 {
		return nil, nil, errors.New("query requires at least one candidate URL")
	}
	if len(candidates) > maxCandidatesPerQuery {
		candidates = candidates[:maxCandidatesPerQuery]
	}
	limit := req.Limit
	if limit <= 0 {
		limit = defaultResultLimit
	}
	if limit > len(candidates) {
		limit = len(candidates)
	}
	timeoutMS := req.TimeoutMS
	if timeoutMS <= 0 {
		timeoutMS = d.cfg.RequestTimeoutMS
	}
	maxBytes := req.MaxBodyBytes
	if maxBytes <= 0 {
		maxBytes = d.cfg.MaxBodyBytes
	}
	queryStarted := time.Now()
	queryCtx, cancel := context.WithTimeout(parent, time.Duration(timeoutMS)*time.Millisecond)
	defer cancel()

	rankedCandidates := d.rankCandidates(req.Query, candidates)
	rankedCandidates = rankedCandidates[:limit]
	resultCh := make(chan FetchResult, len(rankedCandidates))
	enqueued := 0
	for _, candidate := range rankedCandidates {
		work := job{ctx: queryCtx, query: req.Query, candidate: candidate, maxBytes: maxBytes, resultCh: resultCh}
		select {
		case d.jobs <- work:
			enqueued++
		case <-queryCtx.Done():
			return nil, nil, queryCtx.Err()
		}
	}

	results := make([]FetchResult, 0, enqueued)
	for len(results) < enqueued {
		select {
		case result := <-resultCh:
			results = append(results, result)
		case <-queryCtx.Done():
			sortFetchResults(results)
			return results, map[string]any{
				"query":              req.Query,
				"requested":          len(candidates),
				"enqueued":           enqueued,
				"returned":           len(results),
				"timed_out":          true,
				"resident_workers":   d.cfg.Workers,
				"elapsed_ms":         time.Since(queryStarted).Milliseconds(),
				"page_rank_ordering": true,
			}, nil
		}
	}
	sortFetchResults(results)
	d.queriesHandled.Add(1)
	return results, map[string]any{
		"query":              req.Query,
		"requested":          len(candidates),
		"enqueued":           enqueued,
		"returned":           len(results),
		"timed_out":          false,
		"resident_workers":   d.cfg.Workers,
		"elapsed_ms":         time.Since(queryStarted).Milliseconds(),
		"page_rank_ordering": true,
	}, nil
}

func (d *Daemon) rankCandidates(query string, candidates []Candidate) []Candidate {
	terms := queryTerms(query)
	out := append([]Candidate(nil), candidates...)
	sort.SliceStable(out, func(i, j int) bool {
		left := candidatePriority(out[i], terms, d.ranker.Score(out[i].URL))
		right := candidatePriority(out[j], terms, d.ranker.Score(out[j].URL))
		if left != right {
			return left > right
		}
		return out[i].URL < out[j].URL
	})
	return out
}

func candidatePriority(candidate Candidate, terms []string, pageRank float64) float64 {
	text := strings.ToLower(candidate.URL + " " + candidate.Domain + " " + candidate.Title + " " + candidate.Reason)
	score := candidate.Score + (pageRank * 12.0)
	for _, term := range terms {
		if strings.Contains(text, term) {
			score += 1.5
		}
	}
	if candidate.Domain != "" && !strings.HasPrefix(candidate.Domain, "www.") {
		score += 0.1
	}
	return score
}

func (d *Daemon) fetch(ctx context.Context, workerID int, query string, candidate Candidate, maxBytes int64) FetchResult {
	target := strings.TrimSpace(candidate.URL)
	result := FetchResult{
		URL:          target,
		Domain:       candidateDomain(candidate),
		FetchMode:    "resident_go_static",
		WorkerID:     workerID,
		FetchedUnix:  time.Now().Unix(),
		CandidateKey: candidateKey(candidate),
	}
	parsed, err := url.Parse(target)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		result.Error = "invalid http(s) URL"
		return result
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	req.Header.Set("User-Agent", d.cfg.UserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5")
	resp, err := d.client.Do(req)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	defer resp.Body.Close()
	result.StatusCode = resp.StatusCode
	reader := io.LimitReader(resp.Body, maxBytes)
	body, err := io.ReadAll(reader)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	result.Bytes = len(body)
	bodyText := string(body)
	result.Title = firstNonEmpty(candidate.Title, extractTitle(bodyText), parsed.Host)
	result.Snippet = compactText(stripTags(bodyText), 900)
	result.Outlinks = extractLinks(target, bodyText, maxOutlinksPerPage)
	d.ranker.Observe(target, result.Outlinks)
	result.PageRank = d.ranker.Score(target)
	result.Score = result.PageRank*12.0 + lexicalScore(query, result.Title+" "+result.Snippet+" "+target) + statusScore(resp.StatusCode) + candidate.Score
	return result
}

func normalizeCandidates(candidates []Candidate) []Candidate {
	seen := map[string]bool{}
	out := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		candidate.URL = strings.TrimSpace(candidate.URL)
		if candidate.URL == "" || seen[candidate.URL] {
			continue
		}
		seen[candidate.URL] = true
		if candidate.Domain == "" {
			if parsed, err := url.Parse(candidate.URL); err == nil {
				candidate.Domain = strings.ToLower(parsed.Host)
			}
		}
		out = append(out, candidate)
	}
	return out
}

func sortFetchResults(results []FetchResult) {
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].Error == "" && results[j].Error != "" {
			return true
		}
		if results[i].Error != "" && results[j].Error == "" {
			return false
		}
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].URL < results[j].URL
	})
}

func extractTitle(body string) string {
	match := titleRE.FindStringSubmatch(body)
	if len(match) < 2 {
		return ""
	}
	return compactText(stripTags(match[1]), 160)
}

func stripTags(body string) string {
	return html.UnescapeString(tagRE.ReplaceAllString(body, " "))
}

func compactText(text string, limit int) string {
	clean := strings.TrimSpace(spaceRE.ReplaceAllString(text, " "))
	if len(clean) <= limit {
		return clean
	}
	if limit <= 3 {
		return clean[:limit]
	}
	return strings.TrimSpace(clean[:limit-3]) + "..."
}

func extractLinks(baseURL string, body string, limit int) []string {
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, min(limit, maxOutlinksPerPage))
	matches := hrefRE.FindAllStringSubmatch(body, -1)
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		href := strings.TrimSpace(html.UnescapeString(match[1]))
		if href == "" || strings.HasPrefix(href, "#") || strings.HasPrefix(strings.ToLower(href), "javascript:") || strings.HasPrefix(strings.ToLower(href), "mailto:") {
			continue
		}
		parsed, err := url.Parse(href)
		if err != nil {
			continue
		}
		resolved := base.ResolveReference(parsed)
		resolved.Fragment = ""
		if resolved.Scheme != "http" && resolved.Scheme != "https" {
			continue
		}
		normalized := normalizeURL(resolved.String())
		if normalized == "" || seen[normalized] {
			continue
		}
		seen[normalized] = true
		out = append(out, normalized)
		if len(out) >= limit {
			break
		}
	}
	return out
}

func normalizeURL(raw string) string {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	parsed.Host = strings.ToLower(parsed.Host)
	if parsed.Path == "" {
		parsed.Path = "/"
	}
	return parsed.String()
}

func candidateDomain(candidate Candidate) string {
	if candidate.Domain != "" {
		return strings.ToLower(candidate.Domain)
	}
	parsed, err := url.Parse(candidate.URL)
	if err != nil {
		return ""
	}
	return strings.ToLower(parsed.Host)
}

func candidateKey(candidate Candidate) string {
	sum := sha256.Sum256([]byte(candidate.URL + "\x00" + candidate.Title))
	return hex.EncodeToString(sum[:8])
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func statusScore(status int) float64 {
	if status >= 200 && status < 300 {
		return 2.0
	}
	if status >= 300 && status < 400 {
		return 0.75
	}
	if status >= 400 {
		return -4.0
	}
	return 0
}

func lexicalScore(query, text string) float64 {
	terms := queryTerms(query)
	if len(terms) == 0 {
		return 0
	}
	lower := strings.ToLower(text)
	hits := 0
	for _, term := range terms {
		if strings.Contains(lower, term) {
			hits++
		}
	}
	return 6.0 * float64(hits) / float64(len(terms))
}

func queryTerms(query string) []string {
	raw := wordRE.FindAllString(strings.ToLower(query), -1)
	out := make([]string, 0, len(raw))
	seen := map[string]bool{}
	for _, term := range raw {
		if len(term) < 3 || stopTerm(term) || seen[term] {
			continue
		}
		seen[term] = true
		out = append(out, term)
	}
	return out
}

func stopTerm(term string) bool {
	switch term {
	case "what", "who", "where", "when", "why", "how", "the", "and", "for", "with", "from", "about", "tell":
		return true
	default:
		return false
	}
}

type PageRanker struct {
	mu         sync.RWMutex
	outgoing   map[string]map[string]bool
	scores     map[string]float64
	damping    float64
	iterations int
}

func NewPageRanker(damping float64, iterations int) *PageRanker {
	if damping <= 0 || damping >= 1 {
		damping = defaultPageRankDamping
	}
	if iterations <= 0 {
		iterations = defaultPageRankIterations
	}
	return &PageRanker{
		outgoing:   map[string]map[string]bool{},
		scores:     map[string]float64{},
		damping:    damping,
		iterations: iterations,
	}
}

func (p *PageRanker) Observe(source string, outlinks []string) {
	source = normalizeURL(source)
	if source == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.outgoing[source]; !ok {
		p.outgoing[source] = map[string]bool{}
	}
	for _, outlink := range outlinks {
		normalized := normalizeURL(outlink)
		if normalized == "" || normalized == source {
			continue
		}
		p.outgoing[source][normalized] = true
		if _, ok := p.outgoing[normalized]; !ok {
			p.outgoing[normalized] = map[string]bool{}
		}
	}
	p.recomputeLocked()
}

func (p *PageRanker) Score(rawURL string) float64 {
	normalized := normalizeURL(rawURL)
	if normalized == "" {
		return 0
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.scores[normalized]
}

func (p *PageRanker) Telemetry() map[string]any {
	p.mu.RLock()
	defer p.mu.RUnlock()
	edges := 0
	for _, outs := range p.outgoing {
		edges += len(outs)
	}
	return map[string]any{
		"nodes":      len(p.outgoing),
		"edges":      edges,
		"damping":    p.damping,
		"iterations": p.iterations,
	}
}

func (p *PageRanker) recomputeLocked() {
	n := len(p.outgoing)
	if n == 0 {
		return
	}
	nodes := make([]string, 0, n)
	for node := range p.outgoing {
		nodes = append(nodes, node)
	}
	sort.Strings(nodes)
	scores := map[string]float64{}
	initial := 1.0 / float64(n)
	for _, node := range nodes {
		scores[node] = initial
	}
	base := (1.0 - p.damping) / float64(n)
	for iter := 0; iter < p.iterations; iter++ {
		next := map[string]float64{}
		for _, node := range nodes {
			next[node] = base
		}
		sinkMass := 0.0
		for _, source := range nodes {
			outs := p.outgoing[source]
			if len(outs) == 0 {
				sinkMass += scores[source]
				continue
			}
			share := p.damping * scores[source] / float64(len(outs))
			for out := range outs {
				next[out] += share
			}
		}
		if sinkMass > 0 {
			share := p.damping * sinkMass / float64(n)
			for _, node := range nodes {
				next[node] += share
			}
		}
		scores = next
	}
	p.scores = scores
}

func RunJSONL(ctx context.Context, daemon *Daemon, reader io.Reader, writer io.Writer) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	encoder := json.NewEncoder(writer)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var req Request
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			_ = encoder.Encode(Response{Status: "error", Message: err.Error(), ErrorType: "BadJSON"})
			continue
		}
		resp := daemon.Handle(ctx, req)
		if err := encoder.Encode(resp); err != nil {
			return err
		}
		if strings.EqualFold(req.Op, "shutdown") || strings.EqualFold(req.Op, "quit") {
			return nil
		}
	}
	return scanner.Err()
}

func main() {
	cfg := Config{}
	flag.IntVar(&cfg.Workers, "workers", envInt("AXIOM_CRAWL_DAEMON_WORKERS", defaultWorkers), "resident crawler workers")
	flag.Int64Var(&cfg.MaxBodyBytes, "max-body-bytes", envInt64("AXIOM_CRAWL_DAEMON_MAX_BODY_BYTES", defaultMaxBodyBytes), "maximum response bytes per fetch")
	flag.IntVar(&cfg.RequestTimeoutMS, "timeout-ms", envInt("AXIOM_CRAWL_DAEMON_TIMEOUT_MS", defaultRequestTimeoutMS), "default query timeout in milliseconds")
	flag.StringVar(&cfg.UserAgent, "user-agent", envString("AXIOM_CRAWL_DAEMON_USER_AGENT", defaultUserAgent), "HTTP user agent")
	flag.Float64Var(&cfg.PageRankDamping, "pagerank-damping", envFloat("AXIOM_CRAWL_DAEMON_PAGERANK_DAMPING", defaultPageRankDamping), "PageRank damping factor")
	flag.IntVar(&cfg.PageRankIterations, "pagerank-iterations", envInt("AXIOM_CRAWL_DAEMON_PAGERANK_ITERATIONS", defaultPageRankIterations), "PageRank iterations after graph updates")
	flag.Parse()

	daemon := NewDaemon(cfg)
	defer daemon.Shutdown()
	if err := RunJSONL(context.Background(), daemon, os.Stdin, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "crawl daemon error: %v\n", err)
		os.Exit(1)
	}
}

func envString(name string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	return value
}

func envInt(name string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func envInt64(name string, fallback int64) int64 {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func envFloat(name string, fallback float64) float64 {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return fallback
	}
	return parsed
}
