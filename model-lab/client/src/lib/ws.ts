/**
 * WebSocket helper for real-time run progress streaming.
 */

export interface RunEvent {
  ts: string;
  type: string;
  run_id: string;
  [key: string]: unknown;
}

export interface StepProgressEvent extends RunEvent {
  type: 'step_progress';
  step: string;
  progress: number;
  message?: string;
  estimated_remaining_s?: number;
}

export interface StepCompletedEvent extends RunEvent {
  type: 'step_completed';
  step: string;
  duration_ms: number;
  artifacts?: string[];
}

export interface StepFailedEvent extends RunEvent {
  type: 'step_failed';
  step: string;
  error_code: string;
  error_message: string;
  duration_ms: number;
}

export interface RunCompletedEvent extends RunEvent {
  type: 'run_completed';
  status: string;
  total_duration_ms: number;
  steps_completed: number;
  steps_failed: number;
}

export type RunEventHandler = (event: RunEvent) => void;
export type ErrorHandler = (error: Event | string) => void;

/**
 * WebSocket connection for streaming run events.
 */
export class RunEventStream {
  private ws: WebSocket | null = null;
  private runId: string;
  private handlers: RunEventHandler[] = [];
  private errorHandlers: ErrorHandler[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 1000;
  private closed = false;

  constructor(runId: string) {
    this.runId = runId;
  }

  /**
   * Connect to the WebSocket endpoint.
   */
  connect(): void {
    if (this.closed) return;

    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // In development, the API runs on port 8000
    const host = window.location.hostname;
    const port = window.location.port === '5173' || window.location.port === '5174' ? '8000' : window.location.port;
    const wsUrl = `${protocol}//${host}:${port}/api/runs/${this.runId}/ws`;

    console.log(`[WS] Connecting to ${wsUrl}`);

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log(`[WS] Connected to run ${this.runId}`);
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as RunEvent;
          this.handlers.forEach(handler => handler(data));
        } catch (e) {
          console.warn('[WS] Failed to parse message:', event.data);
        }
      };

      this.ws.onclose = (event) => {
        console.log(`[WS] Connection closed: code=${event.code}, reason=${event.reason}`);

        // Normal closure or run completed
        if (event.code === 1000 || this.closed) {
          return;
        }

        // Attempt reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(`[WS] Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts})`);
          setTimeout(() => this.connect(), this.reconnectDelay);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        this.errorHandlers.forEach(handler => handler(error));
      };
    } catch (e) {
      console.error('[WS] Failed to create WebSocket:', e);
      const errorMsg = e instanceof Error ? e.message : String(e);
      this.errorHandlers.forEach(handler => handler(errorMsg));
    }
  }

  /**
   * Add an event handler.
   */
  onEvent(handler: RunEventHandler): void {
    this.handlers.push(handler);
  }

  /**
   * Remove an event handler.
   */
  offEvent(handler: RunEventHandler): void {
    this.handlers = this.handlers.filter(h => h !== handler);
  }

  /**
   * Add an error handler.
   */
  onError(handler: ErrorHandler): void {
    this.errorHandlers.push(handler);
  }

  /**
   * Close the WebSocket connection.
   */
  close(): void {
    this.closed = true;
    if (this.ws) {
      this.ws.close(1000, 'Client closed');
      this.ws = null;
    }
    this.handlers = [];
    this.errorHandlers = [];
  }

  /**
   * Check if the connection is open.
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

/**
 * Create and connect a run event stream.
 */
export function createRunEventStream(runId: string): RunEventStream {
  const stream = new RunEventStream(runId);
  stream.connect();
  return stream;
}

/**
 * React hook for using run event stream (to be used with useEffect).
 *
 * Usage:
 * ```tsx
 * useEffect(() => {
 *   const stream = createRunEventStream(runId);
 *   stream.onEvent((event) => {
 *     // Handle event
 *   });
 *   return () => stream.close();
 * }, [runId]);
 * ```
 */
