package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultProtocolVersion = "2025-11-25"
	defaultServerName      = "axiom-tag"
	defaultServerVersion   = "1.0.5"
)

type JSONRPCRequest struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string           `json:"method"`
	Params  json.RawMessage  `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Result  any              `json:"result,omitempty"`
	Error   *JSONRPCError    `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

type ToolCallParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type ToolDefinition struct {
	Name         string         `json:"name"`
	Title        string         `json:"title,omitempty"`
	Description  string         `json:"description"`
	InputSchema  map[string]any `json:"inputSchema"`
	OutputSchema map[string]any `json:"outputSchema,omitempty"`
	Annotations  map[string]any `json:"annotations,omitempty"`
}

type ToolResult struct {
	Content           []map[string]any `json:"content"`
	StructuredContent map[string]any   `json:"structuredContent,omitempty"`
	IsError           bool             `json:"isError"`
}

type MCPServer struct {
	cfg    ServerConfig
	client *http.Client
	tools  []ToolDefinition
}

type ServerConfig struct {
	Root                 string
	ProtocolVersion      string
	ServerName           string
	ServerTitle          string
	ServerVersion        string
	Python               string
	WorkerTimeout        time.Duration
	AnchorTimeout        time.Duration
	MaxAnchorResults     int
	EnableNetworkAnchors bool
	WikipediaSearchAPI   string
	WikipediaSummaryAPI  string
	GDELTDocAPI          string
	CrossrefWorksAPI     string
	OpenAlexWorksAPI     string
	WaybackCDXAPI        string
	BraveSearchAPI       string
	BraveAPIKeyEnv       string
	UserAgent            string
}

type AnchorBlock struct {
	Source    string         `json:"source"`
	URL       string         `json:"url"`
	Title     string         `json:"title"`
	Text      string         `json:"text"`
	TrustTier string         `json:"trust_tier"`
	Score     float64        `json:"score"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

func main() {
	mode := flag.String("mode", "stdio", "transport mode: stdio")
	configPath := flag.String("config", "", "path to config.toml")
	flag.Parse()

	cfg, err := LoadServerConfig(*configPath)
	if err != nil {
		log.Printf("tag-mcp config warning: %v", err)
	}
	server := NewMCPServer(cfg)
	if *mode != "stdio" {
		log.Printf("unsupported mode %q; only stdio is enabled", *mode)
		os.Exit(2)
	}
	if err := server.ServeStdio(os.Stdin, os.Stdout); err != nil {
		log.Printf("tag-mcp fatal: %v", err)
		os.Exit(1)
	}
}

func NewMCPServer(cfg ServerConfig) *MCPServer {
	return &MCPServer{
		cfg: cfg,
		client: &http.Client{
			Timeout: cfg.AnchorTimeout,
			Transport: &http.Transport{
				Proxy:                 http.ProxyFromEnvironment,
				MaxIdleConns:          64,
				MaxIdleConnsPerHost:   16,
				IdleConnTimeout:       60 * time.Second,
				ResponseHeaderTimeout: cfg.AnchorTimeout,
			},
		},
		tools: BuildTools(),
	}
}

func (s *MCPServer) ServeStdio(in io.Reader, out io.Writer) error {
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	writer := bufio.NewWriter(out)
	defer writer.Flush()

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		response := s.HandleLine(line)
		if response == nil {
			continue
		}
		encoded, err := json.Marshal(response)
		if err != nil {
			log.Printf("failed to marshal MCP response: %v", err)
			continue
		}
		if _, err := writer.Write(append(encoded, '\n')); err != nil {
			return err
		}
		if err := writer.Flush(); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func (s *MCPServer) HandleLine(line string) *JSONRPCResponse {
	var req JSONRPCRequest
	if err := json.Unmarshal([]byte(line), &req); err != nil {
		return ErrorResponse(nil, -32700, "parse error", map[string]any{"detail": err.Error()})
	}
	if req.JSONRPC != "2.0" {
		return ErrorResponse(req.ID, -32600, "invalid JSON-RPC version", nil)
	}
	if req.ID == nil {
		if req.Method != "" {
			if req.Method != "notifications/initialized" && req.Method != "notifications/cancelled" {
				log.Printf("notification: %s", req.Method)
			}
		}
		return nil
	}
	result, rpcErr := s.HandleRequest(context.Background(), req)
	if rpcErr != nil {
		return &JSONRPCResponse{JSONRPC: "2.0", ID: req.ID, Error: rpcErr}
	}
	return &JSONRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: result}
}

func (s *MCPServer) HandleRequest(ctx context.Context, req JSONRPCRequest) (any, *JSONRPCError) {
	switch req.Method {
	case "initialize":
		return s.initializeResult(req.Params), nil
	case "ping":
		return map[string]any{}, nil
	case "tools/list":
		return map[string]any{"tools": s.tools}, nil
	case "tools/call":
		return s.handleToolCall(ctx, req.Params)
	default:
		return nil, &JSONRPCError{Code: -32601, Message: "method not found", Data: map[string]any{"method": req.Method}}
	}
}

func (s *MCPServer) initializeResult(params json.RawMessage) map[string]any {
	version := s.cfg.ProtocolVersion
	var payload map[string]any
	if len(params) > 0 && json.Unmarshal(params, &payload) == nil {
		if requested, ok := payload["protocolVersion"].(string); ok && strings.TrimSpace(requested) != "" {
			version = requested
		}
	}
	return map[string]any{
		"protocolVersion": version,
		"capabilities": map[string]any{
			"tools":   map[string]any{"listChanged": false},
			"logging": map[string]any{},
		},
		"serverInfo": map[string]any{
			"name":        s.cfg.ServerName,
			"title":       s.cfg.ServerTitle,
			"version":     s.cfg.ServerVersion,
			"description": "AXIOM TAG external MCP server: search, query expansion, context injection, VERITAS legitimacy, and anchor acquisition.",
		},
		"instructions": "Call tag.search for full TAG-DIC answers, or anchor.* tools for typed source acquisition.",
	}
}

func (s *MCPServer) handleToolCall(ctx context.Context, raw json.RawMessage) (any, *JSONRPCError) {
	var params ToolCallParams
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, &JSONRPCError{Code: -32602, Message: "invalid tool call params", Data: err.Error()}
	}
	if params.Arguments == nil {
		params.Arguments = map[string]any{}
	}
	result, err := s.CallTool(ctx, params.Name, params.Arguments)
	if err != nil {
		return ToolResult{
			Content: []map[string]any{TextContent(err.Error())},
			StructuredContent: map[string]any{
				"error":      err.Error(),
				"tool":       params.Name,
				"server":     s.cfg.ServerName,
				"occurredAt": time.Now().UTC().Format(time.RFC3339),
			},
			IsError: true,
		}, nil
	}
	return result, nil
}

func (s *MCPServer) CallTool(ctx context.Context, name string, args map[string]any) (ToolResult, error) {
	switch name {
	case "tag.search":
		return s.callPythonTool(ctx, "search", args)
	case "tag.status":
		return s.callPythonTool(ctx, "status", args)
	case "tag.expand":
		return s.callPythonTool(ctx, "expand", args)
	case "tag.veritas":
		return s.callPythonTool(ctx, "veritas", args)
	case "tag.inject_context":
		return s.callPythonTool(ctx, "inject_context", args)
	case "anchor.wikipedia":
		return s.callAnchor(ctx, name, args, s.fetchWikipedia)
	case "anchor.news":
		return s.callAnchor(ctx, name, args, s.fetchNews)
	case "anchor.scholar":
		return s.callAnchor(ctx, name, args, s.fetchScholar)
	case "anchor.wayback":
		return s.callAnchor(ctx, name, args, s.fetchWayback)
	case "anchor.web":
		return s.callAnchor(ctx, name, args, s.fetchBraveWeb)
	default:
		return ToolResult{}, fmt.Errorf("unknown tool: %s", name)
	}
}

func (s *MCPServer) callPythonTool(ctx context.Context, tool string, args map[string]any) (ToolResult, error) {
	timeout := s.cfg.WorkerTimeout
	if t := NumberArg(args, "timeout_seconds", 0); t > 0 {
		timeout = time.Duration(math.Max(1, math.Min(600, t))) * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	body, err := json.Marshal(args)
	if err != nil {
		return ToolResult{}, err
	}
	python := s.cfg.Python
	cmd := exec.CommandContext(ctx, python, "-m", "tag.mcp_worker", tool)
	cmd.Dir = s.cfg.Root
	cmd.Stdin = bytes.NewReader(body)
	cmd.Env = append(os.Environ(),
		"AXIOM_MCP_INTERNAL_CALL=1",
		"PYTHONUNBUFFERED=1",
		"PYTHONPATH="+prependPythonPath(s.cfg.Root, os.Getenv("PYTHONPATH")),
	)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err = cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return ToolResult{}, fmt.Errorf("python worker timed out after %s", timeout)
	}
	if err != nil {
		return ToolResult{}, fmt.Errorf("python worker failed: %w; stderr=%s", err, Trim(stderr.String(), 800))
	}
	var structured map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &structured); err != nil {
		return ToolResult{}, fmt.Errorf("python worker returned invalid JSON: %w; stdout=%s; stderr=%s", err, Trim(stdout.String(), 800), Trim(stderr.String(), 800))
	}
	if stderr.Len() > 0 {
		structured["_stderr_preview"] = Trim(stderr.String(), 800)
	}
	text := StructuredText(structured)
	return ToolResult{
		Content:           []map[string]any{TextContent(text)},
		StructuredContent: structured,
		IsError:           false,
	}, nil
}

func (s *MCPServer) callAnchor(ctx context.Context, tool string, args map[string]any, fn func(context.Context, map[string]any) ([]AnchorBlock, error)) (ToolResult, error) {
	if !s.cfg.EnableNetworkAnchors {
		return ToolResult{}, errors.New("network anchors are disabled by config")
	}
	timeout := s.cfg.AnchorTimeout
	if t := NumberArg(args, "timeout_seconds", 0); t > 0 {
		timeout = time.Duration(math.Max(1, math.Min(120, t))) * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	blocks, err := fn(ctx, args)
	if err != nil {
		return ToolResult{}, err
	}
	sort.SliceStable(blocks, func(i, j int) bool {
		if blocks[i].Score == blocks[j].Score {
			return blocks[i].URL < blocks[j].URL
		}
		return blocks[i].Score > blocks[j].Score
	})
	limit := IntArg(args, "limit", s.cfg.MaxAnchorResults)
	if limit <= 0 || limit > s.cfg.MaxAnchorResults {
		limit = s.cfg.MaxAnchorResults
	}
	if len(blocks) > limit {
		blocks = blocks[:limit]
	}
	structured := map[string]any{
		"tool":        tool,
		"query":       StringArg(args, "query", ""),
		"url":         StringArg(args, "url", ""),
		"blocks":      blocks,
		"count":       len(blocks),
		"server":      s.cfg.ServerName,
		"retrievedAt": time.Now().UTC().Format(time.RFC3339),
	}
	return ToolResult{
		Content:           []map[string]any{TextContent(AnchorBlocksText(blocks))},
		StructuredContent: structured,
		IsError:           false,
	}, nil
}

func (s *MCPServer) fetchWikipedia(ctx context.Context, args map[string]any) ([]AnchorBlock, error) {
	query := AnchorSubjectQuery(StringArg(args, "query", ""))
	if query == "" {
		return nil, errors.New("anchor.wikipedia requires query")
	}
	limit := BoundedInt(IntArg(args, "limit", s.cfg.MaxAnchorResults), 1, s.cfg.MaxAnchorResults)
	searchURL := EndpointWithQuery(s.cfg.WikipediaSearchAPI, map[string]string{
		"action":   "query",
		"list":     "search",
		"srsearch": query,
		"format":   "json",
		"srlimit":  strconv.Itoa(limit),
		"origin":   "*",
	})
	var payload struct {
		Query struct {
			Search []struct {
				PageID  int    `json:"pageid"`
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := s.GetJSON(ctx, searchURL, &payload); err != nil {
		return nil, err
	}
	blocks := make([]AnchorBlock, 0, len(payload.Query.Search))
	var wg sync.WaitGroup
	var mu sync.Mutex
	for idx, item := range payload.Query.Search {
		item := item
		score := 100.0 - float64(idx)
		wg.Add(1)
		go func() {
			defer wg.Done()
			block := AnchorBlock{
				Source:    "wikipedia.org",
				URL:       "https://en.wikipedia.org/wiki/" + url.PathEscape(strings.ReplaceAll(item.Title, " ", "_")),
				Title:     item.Title,
				Text:      CompactText(StripTags(item.Snippet)),
				TrustTier: "wikipedia",
				Score:     score,
				Metadata:  map[string]any{"pageid": item.PageID, "transport": "mcp-stdio", "anchor": "wikipedia"},
			}
			if summary, err := s.fetchWikipediaSummary(ctx, item.Title); err == nil && summary.Text != "" {
				block.URL = summary.URL
				block.Text = summary.Text
				block.Metadata["summary_title"] = summary.Title
			}
			mu.Lock()
			blocks = append(blocks, block)
			mu.Unlock()
		}()
	}
	wg.Wait()
	return blocks, nil
}

type WikiSummary struct {
	Title string
	URL   string
	Text  string
}

func (s *MCPServer) fetchWikipediaSummary(ctx context.Context, title string) (WikiSummary, error) {
	endpoint := strings.TrimRight(s.cfg.WikipediaSummaryAPI, "/") + "/" + url.PathEscape(strings.ReplaceAll(title, " ", "_"))
	var payload struct {
		Title       string `json:"title"`
		Extract     string `json:"extract"`
		ContentURLs struct {
			Desktop struct {
				Page string `json:"page"`
			} `json:"desktop"`
		} `json:"content_urls"`
	}
	if err := s.GetJSON(ctx, endpoint, &payload); err != nil {
		return WikiSummary{}, err
	}
	return WikiSummary{Title: payload.Title, URL: payload.ContentURLs.Desktop.Page, Text: CompactText(payload.Extract)}, nil
}

func (s *MCPServer) fetchNews(ctx context.Context, args map[string]any) ([]AnchorBlock, error) {
	query := strings.TrimSpace(StringArg(args, "query", ""))
	if query == "" {
		return nil, errors.New("anchor.news requires query")
	}
	limit := BoundedInt(IntArg(args, "limit", s.cfg.MaxAnchorResults), 1, s.cfg.MaxAnchorResults)
	newsURL := EndpointWithQuery(s.cfg.GDELTDocAPI, map[string]string{
		"query":      query,
		"mode":       "artlist",
		"format":     "json",
		"maxrecords": strconv.Itoa(limit),
		"sort":       "hybridrel",
	})
	var payload struct {
		Articles []struct {
			URL       string `json:"url"`
			Title     string `json:"title"`
			Domain    string `json:"domain"`
			Source    string `json:"sourcecountry"`
			SeenDate  string `json:"seendate"`
			Language  string `json:"language"`
			SocialImg string `json:"socialimage"`
		} `json:"articles"`
	}
	if err := s.GetJSON(ctx, newsURL, &payload); err != nil {
		return nil, err
	}
	blocks := make([]AnchorBlock, 0, len(payload.Articles))
	for idx, item := range payload.Articles {
		domain := NormalizeDomain(item.Domain)
		if domain == "" {
			domain = NormalizeDomain(item.URL)
		}
		text := CompactText(item.Title)
		if text == "" {
			continue
		}
		blocks = append(blocks, AnchorBlock{
			Source:    domain,
			URL:       item.URL,
			Title:     item.Title,
			Text:      text,
			TrustTier: "news",
			Score:     90.0 - float64(idx),
			Metadata: map[string]any{
				"seen_date": item.SeenDate,
				"language":  item.Language,
				"country":   item.Source,
				"transport": "mcp-stdio",
				"anchor":    "gdelt",
			},
		})
	}
	return blocks, nil
}

func (s *MCPServer) fetchScholar(ctx context.Context, args map[string]any) ([]AnchorBlock, error) {
	query := strings.TrimSpace(StringArg(args, "query", ""))
	if query == "" {
		return nil, errors.New("anchor.scholar requires query")
	}
	limit := BoundedInt(IntArg(args, "limit", s.cfg.MaxAnchorResults), 1, s.cfg.MaxAnchorResults)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var blocks []AnchorBlock
	var firstErr error
	addErr := func(err error) {
		if err == nil {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if firstErr == nil {
			firstErr = err
		}
	}
	wg.Add(2)
	go func() {
		defer wg.Done()
		found, err := s.fetchCrossref(ctx, query, limit)
		addErr(err)
		mu.Lock()
		blocks = append(blocks, found...)
		mu.Unlock()
	}()
	go func() {
		defer wg.Done()
		found, err := s.fetchOpenAlex(ctx, query, limit)
		addErr(err)
		mu.Lock()
		blocks = append(blocks, found...)
		mu.Unlock()
	}()
	wg.Wait()
	if len(blocks) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return blocks, nil
}

func (s *MCPServer) fetchCrossref(ctx context.Context, query string, limit int) ([]AnchorBlock, error) {
	crossrefURL := EndpointWithQuery(s.cfg.CrossrefWorksAPI, map[string]string{
		"query": query,
		"rows":  strconv.Itoa(limit),
	})
	var payload struct {
		Message struct {
			Items []struct {
				Title     []string `json:"title"`
				URL       string   `json:"URL"`
				DOI       string   `json:"DOI"`
				Abstract  string   `json:"abstract"`
				Published struct {
					DateParts [][]int `json:"date-parts"`
				} `json:"published-print"`
			} `json:"items"`
		} `json:"message"`
	}
	if err := s.GetJSON(ctx, crossrefURL, &payload); err != nil {
		return nil, err
	}
	blocks := make([]AnchorBlock, 0, len(payload.Message.Items))
	for idx, item := range payload.Message.Items {
		title := FirstString(item.Title)
		if title == "" {
			continue
		}
		link := item.URL
		if link == "" && item.DOI != "" {
			link = "https://doi.org/" + item.DOI
		}
		text := CompactText(StripTags(item.Abstract))
		if text == "" {
			text = title
		}
		blocks = append(blocks, AnchorBlock{
			Source:    "crossref.org",
			URL:       link,
			Title:     title,
			Text:      text,
			TrustTier: "scholar",
			Score:     80.0 - float64(idx),
			Metadata:  map[string]any{"doi": item.DOI, "transport": "mcp-stdio", "anchor": "crossref"},
		})
	}
	return blocks, nil
}

func (s *MCPServer) fetchOpenAlex(ctx context.Context, query string, limit int) ([]AnchorBlock, error) {
	openAlexURL := EndpointWithQuery(s.cfg.OpenAlexWorksAPI, map[string]string{
		"search":   query,
		"per-page": strconv.Itoa(limit),
	})
	var payload struct {
		Results []struct {
			ID                    string           `json:"id"`
			Title                 string           `json:"title"`
			DisplayName           string           `json:"display_name"`
			DOI                   string           `json:"doi"`
			PublicationYear       int              `json:"publication_year"`
			AbstractInvertedIndex map[string][]int `json:"abstract_inverted_index"`
			PrimaryLocation       map[string]any   `json:"primary_location"`
		} `json:"results"`
	}
	if err := s.GetJSON(ctx, openAlexURL, &payload); err != nil {
		return nil, err
	}
	blocks := make([]AnchorBlock, 0, len(payload.Results))
	for idx, item := range payload.Results {
		title := item.Title
		if title == "" {
			title = item.DisplayName
		}
		if title == "" {
			continue
		}
		link := item.ID
		if item.DOI != "" {
			link = item.DOI
		}
		text := CompactText(ReconstructOpenAlexAbstract(item.AbstractInvertedIndex))
		if text == "" {
			text = title
		}
		blocks = append(blocks, AnchorBlock{
			Source:    "openalex.org",
			URL:       link,
			Title:     title,
			Text:      text,
			TrustTier: "scholar",
			Score:     79.0 - float64(idx),
			Metadata: map[string]any{
				"publication_year": item.PublicationYear,
				"transport":        "mcp-stdio",
				"anchor":           "openalex",
			},
		})
	}
	return blocks, nil
}

func (s *MCPServer) fetchWayback(ctx context.Context, args map[string]any) ([]AnchorBlock, error) {
	target := strings.TrimSpace(StringArg(args, "url", ""))
	if target == "" {
		target = strings.TrimSpace(StringArg(args, "query", ""))
	}
	if target == "" {
		return nil, errors.New("anchor.wayback requires url or query")
	}
	limit := BoundedInt(IntArg(args, "limit", s.cfg.MaxAnchorResults), 1, s.cfg.MaxAnchorResults)
	cdxURL := EndpointWithQuery(s.cfg.WaybackCDXAPI, map[string]string{
		"url":      target,
		"output":   "json",
		"fl":       "timestamp,original,statuscode,mimetype,digest",
		"filter":   "statuscode:200",
		"collapse": "digest",
		"limit":    strconv.Itoa(limit),
	})
	var rows [][]any
	if err := s.GetJSON(ctx, cdxURL, &rows); err != nil {
		return nil, err
	}
	blocks := make([]AnchorBlock, 0, len(rows))
	for idx, row := range rows {
		if idx == 0 || len(row) < 2 {
			continue
		}
		ts := fmt.Sprint(row[0])
		original := fmt.Sprint(row[1])
		status := ""
		if len(row) > 2 {
			status = fmt.Sprint(row[2])
		}
		mimetype := ""
		if len(row) > 3 {
			mimetype = fmt.Sprint(row[3])
		}
		archiveURL := "https://web.archive.org/web/" + ts + "/" + original
		blocks = append(blocks, AnchorBlock{
			Source:    "web.archive.org",
			URL:       archiveURL,
			Title:     "Wayback snapshot of " + original,
			Text:      "Archived snapshot captured at " + ts + " for " + original + ".",
			TrustTier: "wayback",
			Score:     70.0 - float64(idx),
			Metadata: map[string]any{
				"timestamp": ts,
				"original":  original,
				"status":    status,
				"mimetype":  mimetype,
				"transport": "mcp-stdio",
				"anchor":    "wayback",
			},
		})
	}
	return blocks, nil
}

func (s *MCPServer) fetchBraveWeb(ctx context.Context, args map[string]any) ([]AnchorBlock, error) {
	query := strings.TrimSpace(StringArg(args, "query", ""))
	if query == "" {
		return nil, errors.New("anchor.web requires query")
	}
	apiKey := strings.TrimSpace(os.Getenv(s.cfg.BraveAPIKeyEnv))
	if apiKey == "" {
		return nil, fmt.Errorf("anchor.web requires %s", s.cfg.BraveAPIKeyEnv)
	}
	limit := BoundedInt(IntArg(args, "limit", s.cfg.MaxAnchorResults), 1, s.cfg.MaxAnchorResults)
	webURL := EndpointWithQuery(s.cfg.BraveSearchAPI, map[string]string{
		"q":     query,
		"count": strconv.Itoa(limit),
	})
	var payload struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
				PageAge     string `json:"page_age"`
				Profile     struct {
					Name string `json:"name"`
				} `json:"profile"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := s.GetJSONWithHeaders(ctx, webURL, map[string]string{"X-Subscription-Token": apiKey}, &payload); err != nil {
		return nil, err
	}
	blocks := make([]AnchorBlock, 0, len(payload.Web.Results))
	for idx, item := range payload.Web.Results {
		text := CompactText(item.Description)
		if text == "" {
			text = CompactText(item.Title)
		}
		if item.URL == "" || text == "" {
			continue
		}
		domain := NormalizeDomain(item.URL)
		source := domain
		if strings.TrimSpace(item.Profile.Name) != "" {
			source = strings.TrimSpace(item.Profile.Name)
		}
		blocks = append(blocks, AnchorBlock{
			Source:    source,
			URL:       item.URL,
			Title:     CompactText(item.Title),
			Text:      text,
			TrustTier: "web",
			Score:     85.0 - float64(idx),
			Metadata: map[string]any{
				"domain":    domain,
				"page_age":  item.PageAge,
				"transport": "mcp-stdio",
				"anchor":    "brave",
			},
		})
	}
	return blocks, nil
}

func (s *MCPServer) GetJSON(ctx context.Context, endpoint string, target any) error {
	return s.GetJSONWithHeaders(ctx, endpoint, nil, target)
}

func (s *MCPServer) GetJSONWithHeaders(ctx context.Context, endpoint string, headers map[string]string, target any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", s.cfg.UserAgent)
	for key, value := range headers {
		if strings.TrimSpace(key) != "" && strings.TrimSpace(value) != "" {
			req.Header.Set(key, value)
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		preview, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("GET %s returned HTTP %d: %s", endpoint, resp.StatusCode, strings.TrimSpace(string(preview)))
	}
	decoder := json.NewDecoder(io.LimitReader(resp.Body, 8*1024*1024))
	if err := decoder.Decode(target); err != nil {
		return err
	}
	return nil
}

func BuildTools() []ToolDefinition {
	blockArraySchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"blocks": map[string]any{"type": "array", "items": map[string]any{"type": "object"}},
			"count":  map[string]any{"type": "integer"},
		},
	}
	return []ToolDefinition{
		Tool("tag.search", "TAG Search", "Run the full AXIOM TAG search pipeline with swarm, expansion, DIC, and VERITAS.", map[string]any{
			"query": map[string]any{"type": "string"},
			"swarm": map[string]any{"type": "integer", "minimum": 1, "maximum": 500},
			"depth": map[string]any{"type": "integer", "minimum": 1, "maximum": 32},
			"exp":   map[string]any{"type": "integer", "minimum": 0, "maximum": 100},
		}, []string{"query"}, map[string]any{"readOnlyHint": true, "openWorldHint": true}),
		Tool("tag.status", "TAG Status", "Return TAG runtime status and dependency checks.", map[string]any{}, nil, map[string]any{"readOnlyHint": true}),
		Tool("tag.expand", "TAG Query Expansion", "Expand a query through the TAG-DIC GBNF DSL and 100+ query-type taxonomy.", map[string]any{
			"query": map[string]any{"type": "string"},
			"limit": map[string]any{"type": "integer", "minimum": 0, "maximum": 100},
		}, []string{"query"}, map[string]any{"readOnlyHint": true}),
		Tool("tag.veritas", "TAG VERITAS", "Classify low-confidence blocks as CONFIRMED, RUMOR, LEGACY, or CONTESTED.", map[string]any{
			"query":  map[string]any{"type": "string"},
			"blocks": map[string]any{"type": "array", "items": map[string]any{"type": "object"}},
		}, []string{"query", "blocks"}, map[string]any{"readOnlyHint": true}),
		Tool("tag.inject_context", "TAG Direct Context Injection", "Assemble typed DIC slots and a model-ready context injection payload.", map[string]any{
			"query":  map[string]any{"type": "string"},
			"blocks": map[string]any{"type": "array", "items": map[string]any{"type": "object"}},
			"exp":    map[string]any{"type": "integer", "minimum": 0, "maximum": 100},
		}, []string{"query", "blocks"}, map[string]any{"readOnlyHint": true}),
		ToolWithOutput("anchor.wikipedia", "Wikipedia Anchor", "Fetch typed Wikipedia anchor blocks through the MediaWiki and REST summary APIs.", map[string]any{
			"query": map[string]any{"type": "string"},
			"limit": map[string]any{"type": "integer", "minimum": 1, "maximum": 25},
		}, []string{"query"}, map[string]any{"readOnlyHint": true, "openWorldHint": true}, blockArraySchema),
		ToolWithOutput("anchor.news", "News Anchor", "Fetch typed current-news anchor blocks through configured no-key news APIs.", map[string]any{
			"query": map[string]any{"type": "string"},
			"limit": map[string]any{"type": "integer", "minimum": 1, "maximum": 25},
		}, []string{"query"}, map[string]any{"readOnlyHint": true, "openWorldHint": true}, blockArraySchema),
		ToolWithOutput("anchor.scholar", "Scholar Anchor", "Fetch typed academic anchor blocks through Crossref and OpenAlex.", map[string]any{
			"query": map[string]any{"type": "string"},
			"limit": map[string]any{"type": "integer", "minimum": 1, "maximum": 25},
		}, []string{"query"}, map[string]any{"readOnlyHint": true, "openWorldHint": true}, blockArraySchema),
		ToolWithOutput("anchor.wayback", "Wayback Anchor", "Fetch typed archival evidence blocks through the Wayback CDX API.", map[string]any{
			"url":   map[string]any{"type": "string"},
			"query": map[string]any{"type": "string"},
			"limit": map[string]any{"type": "integer", "minimum": 1, "maximum": 25},
		}, nil, map[string]any{"readOnlyHint": true, "openWorldHint": true}, blockArraySchema),
		ToolWithOutput("anchor.web", "Web Index Anchor", "Fetch typed broad-web evidence through Brave Search when BRAVE_SEARCH_API_KEY is configured.", map[string]any{
			"query": map[string]any{"type": "string"},
			"limit": map[string]any{"type": "integer", "minimum": 1, "maximum": 25},
		}, []string{"query"}, map[string]any{"readOnlyHint": true, "openWorldHint": true}, blockArraySchema),
	}
}

func Tool(name, title, description string, props map[string]any, required []string, annotations map[string]any) ToolDefinition {
	return ToolWithOutput(name, title, description, props, required, annotations, nil)
}

func ToolWithOutput(name, title, description string, props map[string]any, required []string, annotations map[string]any, output map[string]any) ToolDefinition {
	if props == nil {
		props = map[string]any{}
	}
	schema := map[string]any{
		"type":                 "object",
		"properties":           props,
		"additionalProperties": true,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return ToolDefinition{
		Name:         name,
		Title:        title,
		Description:  description,
		InputSchema:  schema,
		OutputSchema: output,
		Annotations:  annotations,
	}
}

func LoadServerConfig(path string) (ServerConfig, error) {
	root, err := os.Getwd()
	if err != nil {
		root = "."
	}
	if envRoot := strings.TrimSpace(os.Getenv("AXIOM_ROOT")); envRoot != "" {
		root = envRoot
	}
	if path == "" {
		if envPath := strings.TrimSpace(os.Getenv("AXIOM_CONFIG_TOML")); envPath != "" {
			path = envPath
		} else {
			path = filepath.Join(root, "config.toml")
		}
	}
	cfg := ServerConfig{
		Root:                 root,
		ProtocolVersion:      defaultProtocolVersion,
		ServerName:           defaultServerName,
		ServerTitle:          "AXIOM TAG MCP Server",
		ServerVersion:        defaultServerVersion,
		Python:               ResolvePython(root),
		WorkerTimeout:        120 * time.Second,
		AnchorTimeout:        20 * time.Second,
		MaxAnchorResults:     8,
		EnableNetworkAnchors: true,
		WikipediaSearchAPI:   "https://en.wikipedia.org/w/api.php",
		WikipediaSummaryAPI:  "https://en.wikipedia.org/api/rest_v1/page/summary",
		GDELTDocAPI:          "https://api.gdeltproject.org/api/v2/doc/doc",
		CrossrefWorksAPI:     "https://api.crossref.org/works",
		OpenAlexWorksAPI:     "https://api.openalex.org/works",
		WaybackCDXAPI:        "https://web.archive.org/cdx",
		BraveSearchAPI:       "https://api.search.brave.com/res/v1/web/search",
		BraveAPIKeyEnv:       "BRAVE_SEARCH_API_KEY",
		UserAgent:            "AxiomTAGMCP/1.0.5 (+https://local.axiom.invalid)",
	}
	values, err := ReadTOMLSection(path, "mcp")
	if err != nil {
		return cfg, err
	}
	cfg.ProtocolVersion = StringValue(values, "protocol_version", cfg.ProtocolVersion)
	cfg.ServerName = StringValue(values, "server_name", cfg.ServerName)
	cfg.ServerTitle = StringValue(values, "server_title", cfg.ServerTitle)
	cfg.ServerVersion = StringValue(values, "server_version", cfg.ServerVersion)
	cfg.Python = StringValue(values, "python", cfg.Python)
	if cfg.Python == "auto" {
		cfg.Python = ResolvePython(root)
	}
	cfg.WorkerTimeout = DurationSeconds(values, "worker_timeout_seconds", cfg.WorkerTimeout, 1, 900)
	cfg.AnchorTimeout = DurationSeconds(values, "anchor_timeout_seconds", cfg.AnchorTimeout, 1, 180)
	cfg.MaxAnchorResults = BoundedInt(IntStringValue(values, "max_anchor_results", cfg.MaxAnchorResults), 1, 50)
	cfg.EnableNetworkAnchors = BoolValue(values, "enable_network_anchors", cfg.EnableNetworkAnchors)
	cfg.WikipediaSearchAPI = StringValue(values, "wikipedia_search_api", cfg.WikipediaSearchAPI)
	cfg.WikipediaSummaryAPI = StringValue(values, "wikipedia_summary_api", cfg.WikipediaSummaryAPI)
	cfg.GDELTDocAPI = StringValue(values, "gdelt_doc_api", cfg.GDELTDocAPI)
	cfg.CrossrefWorksAPI = StringValue(values, "crossref_works_api", cfg.CrossrefWorksAPI)
	cfg.OpenAlexWorksAPI = StringValue(values, "openalex_works_api", cfg.OpenAlexWorksAPI)
	cfg.WaybackCDXAPI = StringValue(values, "wayback_cdx_api", cfg.WaybackCDXAPI)
	cfg.BraveSearchAPI = StringValue(values, "brave_search_api", cfg.BraveSearchAPI)
	cfg.BraveAPIKeyEnv = StringValue(values, "brave_api_key_env", cfg.BraveAPIKeyEnv)
	cfg.UserAgent = StringValue(values, "user_agent", cfg.UserAgent)
	return cfg, nil
}

func ReadTOMLSection(path, wanted string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return map[string]string{}, err
	}
	defer file.Close()
	values := map[string]string{}
	section := ""
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.Contains(line, "]") {
			section = strings.TrimSpace(strings.Trim(line, "[]"))
			continue
		}
		if section != wanted {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(StripInlineComment(value))
		values[key] = strings.Trim(value, `"`)
	}
	return values, scanner.Err()
}

func StripInlineComment(value string) string {
	inQuote := false
	escaped := false
	for idx, r := range value {
		if escaped {
			escaped = false
			continue
		}
		if r == '\\' {
			escaped = true
			continue
		}
		if r == '"' {
			inQuote = !inQuote
			continue
		}
		if r == '#' && !inQuote {
			return value[:idx]
		}
	}
	return value
}

func ResolvePython(root string) string {
	if env := strings.TrimSpace(os.Getenv("AXIOM_MCP_PYTHON")); env != "" {
		return env
	}
	candidates := []string{
		filepath.Join(root, ".venv", "bin", "python"),
		filepath.Join(root, ".venv", "Scripts", "python.exe"),
	}
	for _, candidate := range candidates {
		if st, err := os.Stat(candidate); err == nil && !st.IsDir() {
			return candidate
		}
	}
	if path, err := exec.LookPath("python3"); err == nil {
		return path
	}
	if path, err := exec.LookPath("python"); err == nil {
		return path
	}
	return "python"
}

func EndpointWithQuery(base string, params map[string]string) string {
	parsed, err := url.Parse(base)
	if err != nil {
		return base
	}
	q := parsed.Query()
	for key, value := range params {
		q.Set(key, value)
	}
	parsed.RawQuery = q.Encode()
	return parsed.String()
}

func TextContent(text string) map[string]any {
	return map[string]any{"type": "text", "text": text}
}

func ErrorResponse(id *json.RawMessage, code int, message string, data any) *JSONRPCResponse {
	return &JSONRPCResponse{JSONRPC: "2.0", ID: id, Error: &JSONRPCError{Code: code, Message: message, Data: data}}
}

func StructuredText(payload map[string]any) string {
	if ans, ok := payload["answer"].(map[string]any); ok {
		if structured, ok := ans["structured"].(map[string]any); ok {
			if summary, ok := structured["summary"].(string); ok && strings.TrimSpace(summary) != "" {
				return summary
			}
		}
		if text, ok := ans["text"].(string); ok && strings.TrimSpace(text) != "" {
			return text
		}
	}
	if msg, ok := payload["message"].(string); ok && strings.TrimSpace(msg) != "" {
		return msg
	}
	encoded, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Sprint(payload)
	}
	return string(encoded)
}

func AnchorBlocksText(blocks []AnchorBlock) string {
	if len(blocks) == 0 {
		return "No anchor blocks returned."
	}
	var b strings.Builder
	for idx, block := range blocks {
		if idx > 0 {
			b.WriteString("\n\n")
		}
		fmt.Fprintf(&b, "[%d] %s - %s\n%s\n%s", idx+1, block.TrustTier, block.Title, block.URL, Trim(block.Text, 900))
	}
	return b.String()
}

func StringArg(args map[string]any, key, fallback string) string {
	if value, ok := args[key]; ok {
		switch typed := value.(type) {
		case string:
			if strings.TrimSpace(typed) != "" {
				return typed
			}
		case fmt.Stringer:
			if strings.TrimSpace(typed.String()) != "" {
				return typed.String()
			}
		}
	}
	return fallback
}

func NumberArg(args map[string]any, key string, fallback float64) float64 {
	if value, ok := args[key]; ok {
		switch typed := value.(type) {
		case float64:
			return typed
		case int:
			return float64(typed)
		case json.Number:
			parsed, _ := typed.Float64()
			return parsed
		case string:
			parsed, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
			if err == nil {
				return parsed
			}
		}
	}
	return fallback
}

func IntArg(args map[string]any, key string, fallback int) int {
	return int(NumberArg(args, key, float64(fallback)))
}

func StringValue(values map[string]string, key, fallback string) string {
	if value, ok := values[key]; ok && strings.TrimSpace(value) != "" {
		return strings.TrimSpace(value)
	}
	return fallback
}

func IntStringValue(values map[string]string, key string, fallback int) int {
	if value, ok := values[key]; ok {
		if parsed, err := strconv.Atoi(strings.TrimSpace(value)); err == nil {
			return parsed
		}
	}
	return fallback
}

func BoolValue(values map[string]string, key string, fallback bool) bool {
	if value, ok := values[key]; ok {
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "true", "1", "yes", "on":
			return true
		case "false", "0", "no", "off":
			return false
		}
	}
	return fallback
}

func DurationSeconds(values map[string]string, key string, fallback time.Duration, low, high float64) time.Duration {
	value, ok := values[key]
	if !ok {
		return fallback
	}
	parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil {
		return fallback
	}
	parsed = math.Max(low, math.Min(high, parsed))
	return time.Duration(parsed * float64(time.Second))
}

func BoundedInt(value, low, high int) int {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

func FirstString(values []string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func Trim(value string, limit int) string {
	clean := strings.TrimSpace(value)
	if limit <= 0 || len(clean) <= limit {
		return clean
	}
	return strings.TrimSpace(clean[:limit]) + "..."
}

var (
	tagRe        = regexp.MustCompile(`(?is)<[^>]+>`)
	spaceRe      = regexp.MustCompile(`\s+`)
	domainScheme = regexp.MustCompile(`^[a-z][a-z0-9+.-]*://`)
)

func StripTags(value string) string {
	return html.UnescapeString(tagRe.ReplaceAllString(value, " "))
}

func CompactText(value string) string {
	return strings.TrimSpace(spaceRe.ReplaceAllString(html.UnescapeString(value), " "))
}

func NormalizeDomain(raw string) string {
	value := strings.TrimSpace(strings.ToLower(raw))
	if value == "" {
		return ""
	}
	if domainScheme.MatchString(value) {
		parsed, err := url.Parse(value)
		if err == nil {
			value = parsed.Host
		}
	}
	value = strings.Trim(value, ".")
	if slash := strings.IndexAny(value, "/?#"); slash >= 0 {
		value = value[:slash]
	}
	if strings.Contains(value, ":") {
		host, _, found := strings.Cut(value, ":")
		if found {
			value = host
		}
	}
	return value
}

func AnchorSubjectQuery(query string) string {
	value := CompactText(strings.Trim(query, " ?!.\t\r\n"))
	lowered := strings.ToLower(value)
	prefixes := []string{
		"what is ",
		"what are ",
		"who is ",
		"who was ",
		"where is ",
		"when was ",
		"define ",
		"definition of ",
		"overview of ",
	}
	for _, prefix := range prefixes {
		if strings.HasPrefix(lowered, prefix) && len(value) > len(prefix) {
			value = strings.TrimSpace(value[len(prefix):])
			lowered = strings.ToLower(value)
			break
		}
	}
	for _, article := range []string{"a ", "an ", "the "} {
		if strings.HasPrefix(lowered, article) && len(value) > len(article) {
			value = strings.TrimSpace(value[len(article):])
			break
		}
	}
	return value
}

func ReconstructOpenAlexAbstract(index map[string][]int) string {
	if len(index) == 0 {
		return ""
	}
	type pair struct {
		word string
		pos  int
	}
	var pairs []pair
	for word, positions := range index {
		for _, pos := range positions {
			pairs = append(pairs, pair{word: word, pos: pos})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].pos < pairs[j].pos
	})
	words := make([]string, 0, len(pairs))
	for _, p := range pairs {
		words = append(words, p.word)
	}
	return strings.Join(words, " ")
}

func prependPythonPath(root, existing string) string {
	if existing == "" {
		return root
	}
	return root + string(os.PathListSeparator) + existing
}
