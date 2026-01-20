const STALL_THRESHOLD_S = 120;

export function deriveProgressSignal(updatedAt?: string | null): { secondsSinceProgress: number } {
    if (!updatedAt) {
        return { secondsSinceProgress: 0 };
    }
    const ts = Date.parse(updatedAt);
    if (Number.isNaN(ts)) {
        return { secondsSinceProgress: 0 };
    }
    const diff = (Date.now() - ts) / 1000;
    return { secondsSinceProgress: Math.max(0, Math.floor(diff)) };
}

export function isStalled(status: string, secondsSinceProgress: number): boolean {
    if (status !== "RUNNING") {
        return false;
    }
    return secondsSinceProgress > STALL_THRESHOLD_S;
}
