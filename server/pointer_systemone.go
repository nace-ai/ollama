package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/types/model"
)

const pointerSystemOneUnavailable = "model uses a pointer-head decision scorer. Ollama's letter-token System One path does not reproduce its probabilities. Rebuild Ollama against the edlm llama.cpp fork, or start the model's pointer-head server and set OLLAMA_POINTER_RUNNER to its loopback URL."

// handlePointerSystemOne never falls back to letter-token scoring. A local
// GGUF is scored by llama-server POST /v1/systemone. A 404 means that binary
// does not have the route, so the external loopback runner is still accepted.
func (s *Server) handlePointerSystemOne(c *gin.Context, req decision.Request, raw []byte, m *Model) {
	// llama-server scores the original System One JSON. Do not rebuild it.
	body := raw
	if len(body) == 0 {
		var err error
		body, err = json.Marshal(req)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}
	if m != nil && m.isGGUF() {
		runner, _, _, err := s.scheduleRunner(c.Request.Context(), m, []model.Capability{model.CapabilityDecision}, nil, req.KeepAlive, nil)
		if err != nil {
			handleScheduleError(c, req.Model, err)
			return
		}
		status, out, contentType, err := postPointer(c, fmt.Sprintf("http://127.0.0.1:%d/v1/systemone", runner.GetPort()), body)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "pointer-head runner failed"})
			return
		}
		if status != http.StatusNotFound {
			if contentType == "" {
				contentType = "application/json"
			}
			c.Data(status, contentType, out)
			return
		}
	}
	endpoint, err := decision.PointerRunnerEndpoint(os.Getenv("OLLAMA_POINTER_RUNNER"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if endpoint == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": pointerSystemOneUnavailable})
		return
	}
	status, out, contentType, err := postPointer(c, endpoint, body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "pointer-head runner failed"})
		return
	}
	if contentType == "" {
		contentType = "application/json"
	}
	c.Data(status, contentType, out)
}

func postPointer(c *gin.Context, endpoint string, body []byte) (int, []byte, string, error) {
	upstream, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return 0, nil, "", err
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("Accept", "application/json")
	if id := c.GetHeader("x-typesafe-request-id"); id != "" {
		upstream.Header.Set("x-typesafe-request-id", id)
	}
	resp, err := http.DefaultClient.Do(upstream)
	if err != nil {
		return 0, nil, "", err
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, nil, "", err
	}
	if id := resp.Header.Get("x-typesafe-request-id"); id != "" {
		c.Header("x-typesafe-request-id", id)
	}
	return resp.StatusCode, out, resp.Header.Get("Content-Type"), nil
}
