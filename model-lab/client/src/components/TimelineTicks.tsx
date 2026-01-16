import { useEffect, useRef, useState } from 'react';
import type { Segment, Chapter } from '../lib/api';
import { downsampleStride, timeToX, xToTime } from '../lib/timeline';

interface TimelineTicksProps {
    segments: Segment[];
    chapters: Chapter[];
    duration: number;
    audioRef: React.RefObject<HTMLAudioElement | null>;
    onSeek: (time: number) => void;
    height?: number;
}

export function TimelineTicks({ segments, chapters, duration, audioRef, onSeek, height = 40 }: TimelineTicksProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [hoveredChapter, setHoveredChapter] = useState<Chapter | null>(null);
    const [tooltipPos, setTooltipPos] = useState<{ x: number, y: number } | null>(null);

    // Main Draw Loop
    useEffect(() => {
        let animationFrameId: number;

        const render = () => {
            const canvas = canvasRef.current;
            if (!canvas || !audioRef.current) return;

            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const width = canvas.width;
            const h = canvas.height;

            // Clear
            ctx.clearRect(0, 0, width, h);

            // Draw Background Bar
            ctx.fillStyle = '#f3f4f6'; // gray-100
            ctx.fillRect(0, 0, width, h);

            if (duration > 0) {
                // Draw Chapter markers (behind ticks)
                chapters.forEach(ch => {
                    const x1 = timeToX(ch.start_s, width, duration);
                    const x2 = timeToX(ch.end_s, width, duration);
                    const chWidth = Math.max(2, x2 - x1); // Min 2px

                    // Chapter block with slight transparency
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.15)'; // blue-500 with alpha
                    ctx.fillRect(Math.floor(x1), 0, Math.ceil(chWidth), h);

                    // Chapter boundary lines
                    ctx.fillStyle = '#3b82f6'; // blue-500
                    ctx.fillRect(Math.floor(x1), 0, 2, h);
                });

                // Draw Ticks
                ctx.fillStyle = '#d1d5db'; // gray-300
                const stride = downsampleStride(segments.length, width / 2); // 1 tick per 2px max

                for (let i = 0; i < segments.length; i += stride) {
                    const seg = segments[i];
                    const x = timeToX(seg.start_s, width, duration);
                    ctx.fillRect(Math.floor(x), h - 15, 1, 10);
                }

                // Draw Playhead (on top)
                const currentTime = audioRef.current.currentTime;
                const xPlay = timeToX(currentTime, width, duration);

                ctx.fillStyle = '#2563eb'; // blue-600
                ctx.fillRect(Math.floor(xPlay), 0, 2, h);
            }

            animationFrameId = requestAnimationFrame(render);
        };

        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [segments, chapters, duration, audioRef]);

    // Resize Observer (Optional for V1: simple fixed width logic or just rely on CSS width)
    // For V1 we assume parent container width determines internal width
    // We need to handle canvas resolution (DPR)
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const resize = () => {
            canvas.width = canvas.clientWidth;
            canvas.height = height; // Fixed height prop
        };

        resize();
        window.addEventListener('resize', resize);
        return () => window.removeEventListener('resize', resize);
    }, [height]);

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas || duration === 0) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const width = rect.width;
        const time = xToTime(x, width, duration);

        // Find hovered chapter
        const chapter = chapters.find(ch => time >= ch.start_s && time <= ch.end_s);
        setHoveredChapter(chapter || null);

        if (chapter) {
            setTooltipPos({ x: e.clientX, y: e.clientY });
        } else {
            setTooltipPos(null);
        }
    };

    const handleMouseLeave = () => {
        setHoveredChapter(null);
        setTooltipPos(null);
    };

    const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas || duration === 0) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const width = rect.width;

        // Map x to canvas internal width?
        // Wait, e.clientX is relative to viewport. rect.left is viewport relative. 
        // x is pixels within the element.
        // width is element width.

        // We used canvas.width (internal) for drawing.
        // If internal width == clientWidth, then mapping is 1:1.
        // In resize effect above, we set canvas.width = canvas.clientWidth.
        // So safe to use internal width or rect.width.

        const time = xToTime(x, width, duration); // Using visual width

        // Check if clicking on a chapter
        const chapter = chapters.find(ch => time >= ch.start_s && time <= ch.end_s);

        // Seek to chapter start if clicking chapter, otherwise seek to exact time
        onSeek(chapter ? chapter.start_s : time);
    };

    return (
        <div className="relative">
            <canvas
                ref={canvasRef}
                className="w-full cursor-pointer touch-none block"
                style={{ height: `${height}px` }}
                onClick={handleClick}
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
            />

            {hoveredChapter && tooltipPos && (
                <div
                    className="fixed z-50 bg-gray-900 text-white text-xs px-2 py-1 rounded shadow-lg pointer-events-none"
                    style={{
                        left: `${tooltipPos.x + 10}px`,
                        top: `${tooltipPos.y - 30}px`
                    }}
                >
                    <div className="font-semibold">{hoveredChapter.title}</div>
                    <div className="text-gray-300">
                        {formatTime(hoveredChapter.start_s)} - {formatTime(hoveredChapter.end_s)}
                    </div>
                </div>
            )}
        </div>
    );
}

function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
