import { useEffect, useRef, useCallback } from "react";

export interface SSEEvent {
  type: string;
  data: Record<string, unknown>;
}

export function useSSE(onEvent: (event: SSEEvent) => void) {
  const cbRef = useRef(onEvent);
  cbRef.current = onEvent;

  const connect = useCallback(() => {
    const es = new EventSource("/api/events");
    es.onmessage = (e) => {
      try {
        const parsed: SSEEvent = JSON.parse(e.data);
        cbRef.current(parsed);
      } catch {
        // skip malformed events
      }
    };
    es.onerror = () => {
      es.close();
      setTimeout(connect, 3000);
    };
    return es;
  }, []);

  useEffect(() => {
    const es = connect();
    return () => es.close();
  }, [connect]);
}
