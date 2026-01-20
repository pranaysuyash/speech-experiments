import type { RunSummary } from "./api";

function parseDate(value?: string | null): number {
    if (!value) return 0;
    const t = Date.parse(value);
    return Number.isNaN(t) ? 0 : t;
}

export function sortRuns(runs: RunSummary[]): RunSummary[] {
    return [...runs].sort((a, b) => {
        const aStarted = parseDate(a.started_at);
        const bStarted = parseDate(b.started_at);
        if (aStarted !== bStarted) {
            return bStarted - aStarted;
        }
        const aUpdated = parseDate(a.updated_at);
        const bUpdated = parseDate(b.updated_at);
        if (aUpdated !== bUpdated) {
            return bUpdated - aUpdated;
        }
        return a.run_id.localeCompare(b.run_id);
    });
}
