import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function ExperimentsOpenPage() {
    const [expId, setExpId] = useState('');
    const navigate = useNavigate();

    function onOpen(e: React.FormEvent) {
        e.preventDefault();
        if (!expId.trim()) return;
        navigate(`/lab/experiments/${expId.trim()}`);
    }

    return (
        <div className="p-8 max-w-lg mx-auto mt-20 text-center">
            <h1 className="text-2xl font-bold mb-6">Open Experiment</h1>
            <p className="text-gray-600 mb-8">
                Enter an Experiment ID to view its details and compare runs.
            </p>

            <form onSubmit={onOpen} className="flex gap-2">
                <input
                    type="text"
                    value={expId}
                    onChange={(e) => setExpId(e.target.value)}
                    placeholder="exp_..."
                    className="flex-1 border p-2 rounded shadow-sm"
                    autoFocus
                />
                <button
                    type="submit"
                    disabled={!expId.trim()}
                    className="bg-blue-600 text-white px-6 py-2 rounded font-bold hover:bg-blue-700 disabled:opacity-50"
                >
                    Open
                </button>
            </form>
        </div>
    );
}
