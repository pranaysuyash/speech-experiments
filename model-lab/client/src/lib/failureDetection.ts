export type FailureStepResult = {
    step: string;
    isInferred: boolean;
    isUnknown: boolean;
};

const PIPELINE_ORDER = [
    "ingest",
    "asr",
    "diarization",
    "alignment",
    "chapters",
    "summarize_by_speaker",
    "action_items_assignee",
    "bundle",
];

export function getFailureStep(
    failureStep?: string | null,
    stepsCompleted?: string[] | null
): FailureStepResult {
    if (failureStep) {
        return { step: failureStep, isInferred: false, isUnknown: false };
    }

    const completed = stepsCompleted ?? [];
    if (completed.length === 0) {
        return { step: "unknown", isInferred: false, isUnknown: true };
    }

    const next = PIPELINE_ORDER.find((step) => !completed.includes(step));
    if (next) {
        return { step: next, isInferred: true, isUnknown: false };
    }

    const last = completed[completed.length - 1] || "unknown";
    return { step: last, isInferred: true, isUnknown: last === "unknown" };
}
