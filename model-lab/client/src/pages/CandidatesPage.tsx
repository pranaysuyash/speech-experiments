import { useEffect, useState } from 'react';
import { api } from '../lib/api';

interface Candidate {
    candidate_id: string;
    label: string;
    use_case_id: string;
    steps_preset: string;
    expected_artifacts: string[];
}

interface UseCase {
    use_case_id: string;
    title: string;
    description: string;
}

export default function CandidatesPage() {
    const [useCases, setUseCases] = useState<UseCase[]>([]);
    const [candidatesMap, setCandidatesMap] = useState<Record<string, Candidate[]>>({});
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    async function loadData() {
        try {
            const ucs = await api.getUseCases();
            setUseCases(ucs);

            const map: Record<string, Candidate[]> = {};
            await Promise.all(ucs.map(async (uc) => {
                const cands = await api.getCandidatesForUseCase(uc.use_case_id);
                map[uc.use_case_id] = cands;
            }));

            setCandidatesMap(map);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }

    if (loading) return <div className="p-8">Loading candidates...</div>;

    return (
        <div className="p-6 max-w-5xl mx-auto">
            <h1 className="text-2xl font-bold mb-6">Candidates Library</h1>

            {useCases.map((uc) => (
                <section key={uc.use_case_id} className="mb-10">
                    <div className="mb-4">
                        <h2 className="text-xl font-semibold">{uc.title}</h2>
                        <p className="text-gray-600 text-sm">{uc.description}</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {candidatesMap[uc.use_case_id]?.map((cand) => (
                            <div key={cand.candidate_id} className="border rounded-lg p-4 bg-white shadow-sm hover:shadow-md transition-shadow">
                                <div className="flex justify-between items-start mb-2">
                                    <h3 className="font-bold text-lg text-blue-900">{cand.label}</h3>
                                    <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded font-mono">
                                        {cand.candidate_id}
                                    </span>
                                </div>

                                <div className="space-y-2 text-sm">
                                    <div className="flex gap-2">
                                        <span className="font-semibold text-gray-700">Preset:</span>
                                        <span className="bg-blue-50 text-blue-700 px-2 rounded text-xs py-0.5">{cand.steps_preset}</span>
                                    </div>

                                    <div>
                                        <span className="font-semibold text-gray-700 block mb-1">Expected Artifacts:</span>
                                        <div className="flex flex-wrap gap-1">
                                            {cand.expected_artifacts.map((a) => (
                                                <span key={a} className="bg-gray-100 border text-gray-600 px-2 py-0.5 rounded text-xs">
                                                    {a}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}

                        {(!candidatesMap[uc.use_case_id] || candidatesMap[uc.use_case_id].length === 0) && (
                            <div className="text-gray-400 italic text-sm border border-dashed p-4 rounded">No candidates defined.</div>
                        )}
                    </div>
                </section>
            ))}
        </div>
    );
}
