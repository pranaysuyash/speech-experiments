import { useEffect, useState, useRef, type ReactNode } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../lib/api';
import type { RunDetail as RunDetailType, Segment, SearchResult, MeetingPackManifest } from '../lib/api';
import { ArrowLeft, Search, Download, FileText, Star, Trash2, Loader2 } from 'lucide-react';
import { highlightsApi } from '../lib/highlights';
import type { HighlightStore } from '../lib/highlights';
import { getKeyAction, keyboardReducer } from '../lib/keyboard';
import { TimelineTicks } from './TimelineTicks';
import { formatAsSRT, formatAsTXT, downloadText } from '../lib/exporters';

interface RunDetailProps {
    onBack: () => void;
}

const LARGE_RUN_THRESHOLD = 1500;
const SEARCH_DEBOUNCE_MS = 250;

// Simple in-memory cache
const searchCache = new Map<string, SearchResult[]>();

export default function RunDetail({ onBack }: RunDetailProps) {
    const { runId } = useParams<{ runId: string }>();
    const runIdSafe = runId ?? "";
    const [detail, setDetail] = useState<RunDetailType | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [audioUrl, setAudioUrl] = useState("");
    const [highlights, setHighlights] = useState<HighlightStore | null>(null);
    const [activeTab, setActiveTab] = useState<'search' | 'highlights' | 'export'>('search');
    const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
    const [serverSearchResults, setServerSearchResults] = useState<SearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [searchMode, setSearchMode] = useState<'local' | 'server'>('local');
    const [meetingPack, setMeetingPack] = useState<MeetingPackManifest | null>(null);
    const [meetingPackLoading, setMeetingPackLoading] = useState(false);
    const [meetingPackError, setMeetingPackError] = useState<string | null>(null);
    const [previewName, setPreviewName] = useState<string | null>(null);
    const [previewText, setPreviewText] = useState<string | null>(null);
    const [previewCsv, setPreviewCsv] = useState<{ headers: string[]; rows: string[][] } | null>(null);

    const segments = detail?.segments || [];
    const useServerSearch = segments.length > LARGE_RUN_THRESHOLD;

    // Compute filtered segments based on mode
    const filteredSegments = useServerSearch && searchQuery.trim().length >= 2
        ? serverSearchResults.map(r => {
            const seg = segments.find(s => (s as any).id === r.segment_id);
            return seg || { start_s: r.start_s, end_s: r.end_s, text: r.text, speaker: undefined };
        })
        : segments.filter(s => s.text.toLowerCase().includes(searchQuery.toLowerCase()));

    const audioRef = useRef<HTMLAudioElement>(null);
    const searchInputRef = useRef<HTMLInputElement>(null);
    const itemRefs = useRef<(HTMLDivElement | null)[]>([]);
    const abortControllerRef = useRef<AbortController | null>(null);
    const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        if (!runId) return;
        loadDetail();
        setAudioUrl(api.getAudioUrl(runId));
        setHighlights(highlightsApi.get(runId));
        loadMeetingPack();
        // Clear search on run change
        setSearchQuery("");
        setServerSearchResults([]);
        setPreviewName(null);
        setPreviewText(null);
        setPreviewCsv(null);
    }, [runId]);

    // Update search mode based on segment count
    useEffect(() => {
        setSearchMode(useServerSearch ? 'server' : 'local');
    }, [useServerSearch]);

    // Debounced server search
    useEffect(() => {
        if (!runId || !useServerSearch || searchQuery.trim().length < 2) {
            setServerSearchResults([]);
            setIsSearching(false);
            return;
        }

        // Clear previous timer
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
        }

        // Abort previous request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        setIsSearching(true);

        debounceTimerRef.current = setTimeout(async () => {
            const cacheKey = `${runId}:${searchQuery.trim()}`;

            // Check cache
            if (searchCache.has(cacheKey)) {
                setServerSearchResults(searchCache.get(cacheKey)!);
                setIsSearching(false);
                return;
            }

            // Make request
            const controller = new AbortController();
            abortControllerRef.current = controller;

            try {
                const res = await api.searchRun(runId, searchQuery.trim(), 200, controller.signal);
                setServerSearchResults(res.results);
                // Cache results
                searchCache.set(cacheKey, res.results);
            } catch (err) {
                if ((err as any).name !== 'AbortError' && (err as any).name !== 'CanceledError') {
                    console.error('Search failed:', err);
                    setServerSearchResults([]);
                }
            } finally {
                setIsSearching(false);
            }
        }, SEARCH_DEBOUNCE_MS);

        return () => {
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
        };
    }, [searchQuery, runId, useServerSearch]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Global shortcuts
            if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
                e.preventDefault();
                searchInputRef.current?.focus();
                return;
            }

            // Reducer-based nav
            const action = getKeyAction(e);
            if (action) {
                e.preventDefault();
                setSelectedIndex(prev => {
                    const state = { segmentsCount: filteredSegments.length, selectedIndex: prev };
                    const newState = keyboardReducer(state, action);
                    return newState.selectedIndex;
                });
            } else if (e.key === 'Enter' && selectedIndex !== null) {
                const seg = filteredSegments[selectedIndex];
                if (seg) seekTo(seg.start_s);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [filteredSegments.length, selectedIndex]);

    useEffect(() => {
        if (selectedIndex !== null && itemRefs.current[selectedIndex]) {
            itemRefs.current[selectedIndex]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }, [selectedIndex]);

    useEffect(() => {
        setSelectedIndex(null);
    }, [searchQuery, runId]);

    const loadDetail = async () => {
        if (!runId) return;
        try {
            const data = await api.getTranscript(runId);
            setDetail(data);
        } catch (e) {
            console.error(e);
        }
    };

    const loadMeetingPack = async () => {
        if (!runId) return;
        setMeetingPackLoading(true);
        setMeetingPackError(null);
        try {
            const data = await api.getMeetingPackManifest(runId);
            setMeetingPack(data);
        } catch (e: any) {
            // 404 means bundle not generated yet; treat as absent.
            const status = e?.response?.status;
            if (status === 404) {
                setMeetingPack(null);
            } else {
                setMeetingPack(null);
                setMeetingPackError('Failed to load Meeting Pack manifest');
            }
        } finally {
            setMeetingPackLoading(false);
        }
    };

    const loadPreview = async (name: string) => {
        if (!runId) return;
        setPreviewName(name);
        setPreviewText(null);
        setPreviewCsv(null);
        try {
            const url = api.getMeetingPackArtifactPreviewUrl(runId, name, 200_000);
            const res = await fetch(url);
            if (!res.ok) {
                if (res.status === 413) {
                    setPreviewText('Preview too large; download instead.');
                    return;
                }
                throw new Error(`HTTP ${res.status}`);
            }
            const text = await res.text();
            if (name.endsWith('.csv')) {
                const parsed = parseCsv(text);
                setPreviewCsv(parsed);
            } else {
                setPreviewText(text);
            }
        } catch (e) {
            setPreviewText('Failed to load preview');
        }
    };

    const seekTo = (time: number) => {
        if (audioRef.current) {
            audioRef.current.currentTime = time;
            audioRef.current.play();
        }
    };

    const toggleHighlight = (e: React.MouseEvent, seg: Segment) => {
        e.stopPropagation();
        if (!highlights) return;

        // Check if already highlighted (simple check by start/end/text)
        const existing = highlights.items.find(h =>
            h.start_s === seg.start_s && h.end_s === seg.end_s && h.text === seg.text
        );

        if (existing) {
            const updated = highlightsApi.remove(runIdSafe, existing.id);
            setHighlights({ ...updated });
        } else {
            // Add Note? V1 prompt
            // const note = prompt("Add a note (optional):") || "";
            const note = ""; // Skip prompt for speed in V1, rely on sidebar to edit notes later?
            // User asked for "Add note" optional. Let's just star for now.
            const updated = highlightsApi.add(runIdSafe, seg, note);
            setHighlights({ ...updated });
        }
    };

    const exportHighlights = () => {
        if (!highlights || !detail) return;
        const md = highlightsApi.exportMarkdown(highlights, runIdSafe);
        const blob = new Blob([md], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${runIdSafe}_highlights.md`;
        a.click();
    };



    const isHighlighted = (seg: Segment) => {
        return highlights?.items.some(h => h.start_s === seg.start_s && h.end_s === seg.end_s);
    };

    const duration = detail?.segments.length ? detail.segments[detail.segments.length - 1].end_s : 0;

    if (!runId) return <div className="p-8">Missing run id</div>;
    if (!detail) return <div className="p-8">Loading detail...</div>;

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white border-b py-3 px-6 flex items-center justify-between shadow-sm z-10">
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className="p-2 hover:bg-gray-100 rounded-full">
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <h2 className="font-bold text-lg">{detail.run_id}</h2>
                        <div className="text-xs text-gray-500">Analyst Console</div>
                    </div>
                </div>

                <div className="flex gap-2">
                    <a href={api.getMeetingPackZipUrl(runId)} target="_blank" rel="noreferrer" className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                        <Download size={16} /> Meeting Pack ZIP
                    </a>
                </div>
            </header>

            {/* Main Content */}
            <div className="flex flex-1 overflow-hidden">
                {/* Transcript Panel */}
                <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto relative">
                    <div className="sticky top-0 z-10 bg-white/95 backdrop-blur pb-4 border-b mb-6">
                        <TimelineTicks
                            segments={segments}
                            chapters={detail?.chapters || []}
                            duration={duration}
                            audioRef={audioRef}
                            onSeek={seekTo}
                            height={48}
                        />
                    </div>

                    <div className="space-y-6">
                        {filteredSegments.length === 0 && (
                            <div className="text-center text-gray-400 mt-10">No matches found</div>
                        )}

                        {filteredSegments.map((seg, idx) => {
                            const highlighted = isHighlighted(seg);
                            const selected = idx === selectedIndex;
                            return (
                                <div
                                    key={idx}
                                    ref={el => { itemRefs.current[idx] = el; }}
                                    className={`group p-2 rounded cursor-pointer transition-colors border-l-4 
                                        ${selected ? 'bg-blue-50 border-blue-500 ring-1 ring-blue-200' : 'border-transparent hover:bg-blue-50'}
                                        ${highlighted ? 'bg-yellow-50 border-yellow-200' : ''}`}
                                    onClick={() => {
                                        seekTo(seg.start_s);
                                        setSelectedIndex(idx);
                                    }}
                                >
                                    <div className="flex gap-4">
                                        <div className="w-24 flex-shrink-0 text-xs text-gray-400 font-mono mt-1 select-none flex flex-col items-end gap-1">
                                            <span>{formatTime(seg.start_s)}</span>
                                            <div
                                                className={`p-1 rounded hover:bg-yellow-200 ${highlighted ? 'text-yellow-500' : 'text-gray-200 group-hover:text-gray-400'}`}
                                                onClick={(e) => toggleHighlight(e, seg)}
                                                title="Toggle Highlight"
                                            >
                                                <Star size={14} fill={highlighted ? "currentColor" : "none"} />
                                            </div>
                                        </div>
                                        <div className="flex-1">
                                            {seg.speaker && <div className="font-bold text-xs text-gray-600 mb-0.5">{seg.speaker}</div>}
                                            <p className={`text-gray-800 leading-relaxed text-lg ${highlighted ? 'font-medium' : ''}`}>
                                                {highlightText(seg.text, searchQuery)}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>

                {/* Right Sidebar */}
                <div className="w-80 bg-white border-l flex flex-col">
                    {/* Audio Player */}
                    <div className="p-4 border-b bg-gray-50">
                        <h3 className="font-semibold mb-2 text-xs text-uppercase text-gray-500 tracking-wider">AUDIO</h3>
                        <audio ref={audioRef} controls src={audioUrl} className="w-full h-8" />
                    </div>

                    {/* Tabs */}
                    <div className="flex border-b">
                        <button
                            className={`flex-1 py-3 text-sm font-medium ${activeTab === 'search' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                            onClick={() => setActiveTab('search')}
                        >
                            Search
                        </button>
                        <button
                            className={`flex-1 py-3 text-sm font-medium ${activeTab === 'highlights' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                            onClick={() => setActiveTab('highlights')}
                        >
                            Highlights ({highlights?.items.length || 0})
                        </button>
                        <button
                            className={`flex-1 py-3 text-sm font-medium ${activeTab === 'export' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                            onClick={() => setActiveTab('export')}
                        >
                            Export
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="flex-1 overflow-y-auto p-4 content-start">
                        {activeTab === 'search' && (
                            <div>
                                <div className="relative">
                                    <Search className="absolute left-3 top-2.5 text-gray-400" size={18} />
                                    {isSearching && (
                                        <Loader2 className="absolute right-3 top-2.5 text-blue-500 animate-spin" size={18} />
                                    )}
                                    <input
                                        ref={searchInputRef}
                                        type="text"
                                        placeholder="Search transcript..."
                                        className="w-full pl-10 pr-10 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                        value={searchQuery}
                                        onChange={e => setSearchQuery(e.target.value)}
                                    />
                                </div>
                                {searchQuery.trim() && (
                                    <div className="text-xs text-gray-500 mt-1 px-1">
                                        {searchMode === 'server' ? 'Searching on server' : 'Searching locally'}
                                    </div>
                                )}
                            </div>
                        )}

                        {activeTab === 'highlights' && highlights && (
                            <div className="space-y-4">
                                {highlights.items.length === 0 ? (
                                    <div className="text-gray-400 text-sm text-center mt-10">
                                        Star segments to add them here.
                                    </div>
                                ) : (
                                    <>
                                        <button
                                            onClick={exportHighlights}
                                            className="w-full py-2 px-4 bg-gray-800 text-white rounded text-sm hover:bg-gray-700 flex items-center justify-center gap-2 mb-4"
                                        >
                                            <FileText size={14} /> Export Markdown
                                        </button>
                                        {highlights.items.map(item => (
                                            <div key={item.id} className="text-sm p-3 bg-yellow-50 rounded border border-yellow-100 group relative">
                                                <div className="flex justify-between items-start mb-1 text-xs text-gray-500">
                                                    <span>{formatTime(item.start_s)} - {formatTime(item.end_s)}</span>
                                                    <button
                                                        className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                                                        onClick={() => {
                                                            const updated = highlightsApi.remove(runId, item.id);
                                                            setHighlights({ ...updated });
                                                        }}
                                                    >
                                                        <Trash2 size={12} />
                                                    </button>
                                                </div>
                                                <p className="line-clamp-3 mb-1" onClick={() => seekTo(item.start_s)}>{item.text}</p>
                                                {item.note && <div className="text-xs text-gray-600 italic border-l-2 border-yellow-300 pl-2 mt-1">{item.note}</div>}
                                            </div>
                                        ))}
                                    </>
                                )}
                            </div>
                        )}

                        {activeTab === 'export' && (
                            <div className="space-y-3">
                                <div className="text-sm text-gray-600 mb-4">
                                    Download run artifacts and formatted transcripts.
                                </div>

                                {/* Meeting Pack */}
                                <div className="border rounded-lg p-3 bg-gray-50">
                                    <div className="font-semibold text-sm mb-2">Meeting Pack</div>
                                    {meetingPackLoading && (
                                        <div className="text-xs text-gray-500">Loading…</div>
                                    )}
                                    {!meetingPackLoading && meetingPackError && (
                                        <div className="text-xs text-red-600">{meetingPackError}</div>
                                    )}
                                    {!meetingPackLoading && !meetingPackError && !meetingPack && (
                                        <div className="text-xs text-gray-500">Not available yet for this run.</div>
                                    )}
                                    {!meetingPackLoading && meetingPack && (
                                        <div className="space-y-2">
                                            <div className="flex gap-2">
                                                <a
                                                    href={api.getMeetingPackZipUrl(runId)}
                                                    download
                                                    className="flex-1 py-2 px-3 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 flex items-center justify-center gap-2 transition-colors"
                                                >
                                                    <Download size={14} /> Download All (ZIP)
                                                </a>
                                                <a
                                                    href={api.getSessionBundleZipUrl(runId)}
                                                    download
                                                    className="flex-1 py-2 px-3 bg-gray-700 text-white rounded text-xs hover:bg-gray-600 flex items-center justify-center gap-2 transition-colors"
                                                >
                                                    <Download size={14} /> Session ZIP
                                                </a>
                                            </div>

                                            <div className="space-y-1">
                                                {meetingPack.artifacts.map((a) => (
                                                    <div key={a.name} className="flex items-center justify-between gap-2 text-xs">
                                                        <div className="min-w-0">
                                                            <div className="font-mono truncate">{a.name}</div>
                                                            <div className="text-gray-500">{humanBytes(a.bytes)}</div>
                                                        </div>
                                                        <div className="flex gap-1">
                                                            {(a.name === 'summary.md' || a.name === 'decisions.md' || a.name === 'action_items.csv') && (
                                                                <button
                                                                    onClick={() => loadPreview(a.name)}
                                                                    className="px-2 py-1 rounded bg-white border hover:bg-gray-100"
                                                                >
                                                                    Preview
                                                                </button>
                                                            )}
                                                            <a
                                                                href={api.getMeetingPackArtifactUrl(runId, a.name)}
                                                                download
                                                                className="px-2 py-1 rounded bg-white border hover:bg-gray-100"
                                                            >
                                                                Download
                                                            </a>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>

                                            {meetingPack.absent?.length > 0 && (
                                                <div className="pt-2 border-t">
                                                    <div className="text-xs font-semibold text-gray-600 mb-1">Missing</div>
                                                    <div className="space-y-1">
                                                        {meetingPack.absent.map((m) => (
                                                            <div key={m.name} className="text-xs text-gray-500">
                                                                <span className="font-mono">{m.name}</span>: {m.reason}
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>

                                {(previewText || previewCsv) && (
                                    <div className="border rounded-lg p-3 bg-white">
                                        <div className="text-xs font-semibold text-gray-600 mb-2">Preview: <span className="font-mono">{previewName}</span></div>
                                        {previewText && (
                                            <div className="text-sm text-gray-800 space-y-2">
                                                {renderMarkdown(previewText)}
                                            </div>
                                        )}
                                        {previewCsv && (
                                            <div className="overflow-auto">
                                                <table className="min-w-full text-xs">
                                                    <thead>
                                                        <tr className="text-left border-b">
                                                            {previewCsv.headers.map((h) => (
                                                                <th key={h} className="py-1 pr-3 font-semibold">{h}</th>
                                                            ))}
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {previewCsv.rows.slice(0, 20).map((row, i) => (
                                                            <tr key={i} className="border-b last:border-b-0">
                                                                {row.map((cell, j) => (
                                                                    <td key={j} className="py-1 pr-3 whitespace-pre-wrap">{cell}</td>
                                                                ))}
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                                {previewCsv.rows.length > 20 && (
                                                    <div className="text-[11px] text-gray-500 mt-2">Showing first 20 rows.</div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Transcript SRT */}
                                <button
                                    onClick={() => {
                                        const srt = formatAsSRT(segments);
                                        downloadText(srt, `run_${runId}.srt`, 'text/plain');
                                    }}
                                    disabled={segments.length === 0}
                                    className="w-full py-3 px-4 bg-gray-700 text-white rounded text-sm hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
                                >
                                    <FileText size={16} /> Download Transcript (SRT)
                                </button>

                                {/* Transcript TXT */}
                                <button
                                    onClick={() => {
                                        const txt = formatAsTXT(segments, runId);
                                        downloadText(txt, `run_${runId}.txt`, 'text/plain');
                                    }}
                                    disabled={segments.length === 0}
                                    className="w-full py-3 px-4 bg-gray-700 text-white rounded text-sm hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
                                >
                                    <FileText size={16} /> Download Transcript (TXT)
                                </button>

                                {/* Highlights MD */}
                                <button
                                    onClick={exportHighlights}
                                    disabled={!highlights || highlights.items.length === 0}
                                    className="w-full py-3 px-4 bg-yellow-600 text-white rounded text-sm hover:bg-yellow-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
                                >
                                    <Star size={16} /> Download Highlights (MD)
                                </button>

                                {(!highlights || highlights.items.length === 0) && (
                                    <div className="text-xs text-gray-500 text-center mt-2">
                                        Star segments to enable highlights export.
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

// Helpers
function formatTime(s: number) {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function highlightText(text: string, query: string) {
    if (!query) return text;
    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase()
            ? <span key={i} className="bg-yellow-200 rounded-sm">{part}</span>
            : part
    );
}

function humanBytes(bytes: number) {
    if (!Number.isFinite(bytes)) return '--';
    const units = ['B', 'KB', 'MB', 'GB'];
    let b = bytes;
    let i = 0;
    while (b >= 1024 && i < units.length - 1) {
        b /= 1024;
        i++;
    }
    const n = i === 0 ? b.toFixed(0) : b.toFixed(1);
    return `${n} ${units[i]}`;
}

function parseCsv(input: string): { headers: string[]; rows: string[][] } {
    const rows: string[][] = [];
    let row: string[] = [];
    let field = '';
    let inQuotes = false;

    const pushField = () => {
        row.push(field);
        field = '';
    };
    const pushRow = () => {
        // Skip trailing empty row from final newline
        if (row.length === 1 && row[0] === '' && rows.length === 0) return;
        rows.push(row);
        row = [];
    };

    for (let i = 0; i < input.length; i++) {
        const c = input[i];
        if (inQuotes) {
            if (c === '"') {
                const next = input[i + 1];
                if (next === '"') {
                    field += '"';
                    i++;
                } else {
                    inQuotes = false;
                }
            } else {
                field += c;
            }
            continue;
        }

        if (c === '"') {
            inQuotes = true;
            continue;
        }
        if (c === ',') {
            pushField();
            continue;
        }
        if (c === '\n') {
            pushField();
            pushRow();
            continue;
        }
        if (c === '\r') {
            continue;
        }
        field += c;
    }
    pushField();
    pushRow();

    const headers = rows.length ? rows[0] : [];
    const dataRows = rows.length ? rows.slice(1) : [];
    return { headers, rows: dataRows };
}

function renderMarkdown(md: string) {
    const lines = md.split('\n');
    const out: ReactNode[] = [];
    let list: string[] = [];

    const flushList = (keyPrefix: string) => {
        if (list.length === 0) return;
        out.push(
            <ul key={`${keyPrefix}-ul`} className="list-disc ml-5 space-y-1">
                {list.map((t, i) => <li key={i}>{t}</li>)}
            </ul>
        );
        list = [];
    };

    let key = 0;
    for (const raw of lines) {
        const line = raw.replace(/\s+$/, '');
        const h3 = line.match(/^###\s+(.*)$/);
        const h2 = line.match(/^##\s+(.*)$/);
        const h1 = line.match(/^#\s+(.*)$/);
        const li = line.match(/^\s*[-*]\s+(.*)$/);

        if (h1 || h2 || h3) {
            flushList(String(key));
            const text = (h1?.[1] || h2?.[1] || h3?.[1] || '').trim();
            if (h1) out.push(<h1 key={key++} className="text-base font-bold">{text}</h1>);
            else if (h2) out.push(<h2 key={key++} className="text-sm font-semibold mt-2">{text}</h2>);
            else out.push(<h3 key={key++} className="text-xs font-semibold mt-2 text-gray-700">{text}</h3>);
            continue;
        }

        if (li) {
            list.push(li[1].trim());
            continue;
        }

        if (!line.trim()) {
            flushList(String(key));
            continue;
        }

        flushList(String(key));
        out.push(<p key={key++} className="text-sm leading-relaxed whitespace-pre-wrap">{line}</p>);
    }
    flushList(String(key));
    return <>{out}</>;
}
