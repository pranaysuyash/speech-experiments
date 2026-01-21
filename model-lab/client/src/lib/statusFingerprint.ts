/**
 * Status Fingerprint Logic
 * 
 * Computes a stable fingerprint string from a Run Status object.
 * Used to ensure consistency between different API entry points (Runs Index vs Run Detail vs Findings).
 */

export interface HelperStatusPayload {
    run_id: string;
    status: string;
    current_step: string | null;
    steps_completed: string[];
    failed_step: string | null;
    error_message: string | null;
    snapshot_source: string | null;
    manifest_mtime: string | null;
    manifest_schema_version?: number;
}

function stableStringify(value: unknown): string {
    if (value === null || typeof value !== "object") return JSON.stringify(value);

    if (Array.isArray(value)) {
        return `[${value.map(stableStringify).join(",")}]`;
    }

    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`).join(",")}}`;
}

export function deriveStatusFingerprint(payload: HelperStatusPayload): string {
    // 1. Gather fields in strict order to build a canonical object
    const canonical: any = {};
    
    canonical.current_step = payload.current_step || null;
    canonical.error_message = payload.error_message ? payload.error_message.trim() : null;
    canonical.failed_step = payload.failed_step || null;
    canonical.manifest_mtime = payload.manifest_mtime || null;
    canonical.manifest_schema_version = payload.manifest_schema_version || 1;
    canonical.run_id = payload.run_id;
    canonical.snapshot_source = payload.snapshot_source || "unknown";
    canonical.status = payload.status;
    
    // Normalize array: sorted and original deduplicated
    const sortedSteps = Array.from(new Set(payload.steps_completed || [])).sort();
    canonical.steps_completed = sortedSteps;
    
    return stableStringify(canonical);
}

export async function computeFingerprintHash(raw: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(raw);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}
