import { useEffect, useState, useRef, type ReactNode } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../lib/api';
import type { RunDetail as RunDetailType, Segment, SearchResult, MeetingPackManifest, ResultSummary } from '../lib/api';

// ... (imports)

export default function RunDetail({ onBack }: RunDetailProps) {
    const { runId } = useParams<{ runId: string }>();
    const runIdSafe = runId ?? "";
    const [status, setStatus] = useState<any>(null); // TODO: type this proper
    const [detail, setDetail] = useState<RunDetailType | null>(null);
    const [result, setResult] = useState<ResultSummary | null>(null);
    const [searchQuery, setSearchQuery] = useState("");

    // ...

    // Poll for status until terminal
    useEffect(() => {
        if (!runId) return;
        let pollTimer: ReturnType<typeof setInterval>;

        const checkStatus = async () => {
            try {
                const s = await api.getRunStatus(runId);
                setStatus(s);

                // Terminal State Handling
                if (['COMPLETED', 'FAILED', 'STALE'].includes(s.status)) {
                    // Stop polling
                    if (pollTimer) clearInterval(pollTimer);

                    // Fetch semantic results
                    try {
                        const res = await api.getRunResults(runId);
                        setResult(res);

                        // If we have useful artifacts (COMPLETED or PARTIAL), load details
                        const shouldLoadArtifacts = s.status === 'COMPLETED' || (s.status === 'FAILED' && res.quality_flags.is_partial);

                        if (shouldLoadArtifacts && !detail) {
                            loadDetail();
                            setAudioUrl(api.getAudioUrl(runId));
                            setHighlights(highlightsApi.get(runId));
                            loadMeetingPack();
                        }
                    } catch (err) {
                        console.error("Failed to load results", err);
                    }
                }
            } catch (e) {
                console.error("Failed to get status", e);
            }
        };

        checkStatus();
        pollTimer = setInterval(checkStatus, 2000); // 2s polling

        return () => clearInterval(pollTimer);
    }, [runId]);

    // Clear search on run change
    useEffect(() => {
        setSearchQuery("");
        setServerSearchResults([]);
        setPreviewName(null);
        setPreviewText(null);
        setPreviewCsv(null);
        setStatus(null);
        setDetail(null);
    }, [runId]);

    // ... search and keyboard effects omitted for brevity, they remain ...

    const loadDetail = async () => {
        if (!runId) return;
        try {
            const data = await api.getTranscript(runId);
            setDetail(data);
        } catch (e) {
            console.error(e);
        }
    };

    // ... loadMeetingPack, loadPreview, seekTo, toggleHighlight, exportHighlights ...
    // ... kept as is ...

    if (!runId) return <div className="p-8">Missing run id</div>;

    if (!status) return <div className="p-8 flex items-center gap-2"><Loader2 className="animate-spin" /> Loading run status...</div>;

    if (status.status === 'QUEUED' || status.status === 'RUNNING') {
        return (
            <div className="p-8 max-w-2xl mx-auto text-center mt-20">
                <Loader2 className="animate-spin mx-auto mb-4 text-blue-600" size={48} />
                <h2 className="text-xl font-bold mb-2">Run is {status.status}</h2>
                <p className="text-gray-600">Current Step: {status.current_step || 'Initializing...'}</p>
                <div className="mt-8 text-sm text-gray-500 font-mono">Run ID: {runId}</div>
                <button onClick={onBack} className="mt-8 px-4 py-2 border rounded hover:bg-gray-50">Back to List</button>
            </div>
        );
    }

    const isFailed = status.status === 'FAILED' || status.status === 'STALE';
    const showPartial = isFailed && result?.quality_flags.is_partial;

    if (isFailed && !showPartial) {
        return (
            <div className="p-8 max-w-2xl mx-auto text-center mt-20">
                <div className="mx-auto mb-4 w-12 h-12 bg-red-100 text-red-600 rounded-full flex items-center justify-center">
                    <Trash2 size={24} />
                </div>
                <h2 className="text-xl font-bold mb-2 text-red-700">Run Failed</h2>
                <p className="text-gray-800 font-medium">{status.error_code}</p>
                <p className="text-gray-600 mt-2">{status.error_message}</p>
                <div className="mt-8 text-sm text-gray-400 font-mono">Run ID: {runId}</div>
                <button onClick={onBack} className="mt-8 px-4 py-2 border rounded hover:bg-gray-50">Back to List</button>
            </div>
        );
    }

    if (!detail) return <div className="p-8 flex items-center gap-2"><Loader2 className="animate-spin" /> Loading transcript...</div>;

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white border-b py-3 px-6 flex items-center justify-between shadow-sm z-10">
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className="p-2 hover:bg-gray-100 rounded-full">
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <div className="flex items-center gap-2">
                            <h2 className="font-bold text-lg">{detail.run_id}</h2>
                            {result?.quality_flags.is_partial && (
                                <span className="px-2 py-0.5 rounded text-xs font-semibold bg-orange-100 text-orange-700 border border-orange-200">
                                    PARTIAL
                                </span>
                            )}
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-500 mt-1 font-mono">
                            {result?.metrics && (
                                <>
                                    {result.metrics.duration_s && (
                                        <span>⏱ {result.metrics.duration_s.toFixed(1)}s</span>
                                    )}
                                    {result.metrics.word_count !== undefined && (
                                        <span>📝 {result.metrics.word_count} words</span>
                                    )}
                                    {result.metrics.confidence_avg && (
                                        <span>🎯 {(result.metrics.confidence_avg * 100).toFixed(1)}%</span>
                                    )}
                                </>
                            )}
                            {!result && <span>Analyst Console</span>}
                        </div>
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
