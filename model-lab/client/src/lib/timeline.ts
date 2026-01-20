export function downsampleStride(total: number, maxPoints: number): number {
    if (total <= 0) return 1;
    const stride = Math.ceil(total / Math.max(1, maxPoints));
    return Math.max(1, stride);
}

export function timeToX(time: number, width: number, duration: number): number {
    if (duration <= 0 || width <= 0) return 0;
    const ratio = Math.min(1, Math.max(0, time / duration));
    return ratio * width;
}

export function xToTime(x: number, width: number, duration: number): number {
    if (width <= 0 || duration <= 0) return 0;
    const ratio = Math.min(1, Math.max(0, x / width));
    return ratio * duration;
}
