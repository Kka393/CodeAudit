const API_BASE = import.meta.env.VITE_API_BASE || "";

/**
 * Check backend health status
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Run full 4-class multi-view prediction pipeline
 */
export async function predict(code, language, confidenceThreshold = null) {
  const body = { code, language };
  if (confidenceThreshold !== null) {
    body.confidence_threshold = confidenceThreshold;
  }

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Prediction failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Start streaming multi-round LLM debate over SSE
 */
export function startDebateStream(code, aiProb, maxRounds, onEvent, onError, onDone) {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${API_BASE}/debate/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({
          code,
          ai_prob: Number(aiProb),
          max_rounds: Number(maxRounds || 3),
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Debate request failed: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const lines = part.split("\n");
          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data:")) {
              const jsonStr = trimmed.slice(5).trim();
              if (jsonStr) {
                try {
                  const event = JSON.parse(jsonStr);
                  onEvent(event);
                  if (event.event_type === "done" || event.event_type === "error") {
                    if (onDone) onDone(event);
                    return;
                  }
                } catch (e) {
                  console.error("Failed to parse debate SSE line:", jsonStr, e);
                }
              }
            }
          }
        }
      }

      if (onDone) onDone({ event_type: "done" });
    } catch (err) {
      if (err.name === "AbortError") return;
      console.error("Debate stream error:", err);
      if (onError) onError(err);
    }
  })();

  // Return cancel handle
  return () => {
    controller.abort();
  };
}

/**
 * Generate source-code line-level attribution for 4-class prediction
 */
export async function explainPrediction(code, language, targetClass = null) {
  const body = { code, language };
  if (targetClass) {
    body.target_class = targetClass;
  }

  const response = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Attribution failed with status ${response.status}`);
  }

  return response.json();
}
