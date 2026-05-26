import { useEffect, useRef, useCallback } from "react";

export interface SSEEvent {
  type: string;
  data: Record<string, unknown>;
}

export function useSSE(onEvent: (event: SSEEvent) => void) {
  const cbRef = useRef(onEvent);
  cbRef.current = onEvent;

  const connect = useCallback(() => {
    let es: EventSource | null = null;
    try {
      es = new EventSource("/api/events");
      es.onmessage = (e) => {
        try {
          const parsed: SSEEvent = JSON.parse(e.data);
          cbRef.current(parsed);
        } catch {
          // skip malformed events
        }
      };
      es.onerror = () => {
        es?.close();
        // In static mode, don't reconnect aggressively
        setTimeout(connect, 30000);
      };
    } catch {
      // SSE not available (static hosting)
    }
    return es;
  }, []);

  useEffect(() => {
    const es = connect();
    return () => es?.close();
  }, [connect]);
}
