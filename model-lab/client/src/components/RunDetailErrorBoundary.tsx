import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertCircle } from 'lucide-react';

interface Props {
    children: ReactNode;
    runId?: string;
    status?: string;
    onBack?: () => void;
}

interface State {
    hasError: boolean;
    error?: Error;
}

export class RunDetailErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('RunDetail Error:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="p-8 max-w-2xl mx-auto text-center mt-20">
                    <div className="mx-auto mb-4 w-12 h-12 bg-red-100 text-red-600 rounded-full flex items-center justify-center">
                        <AlertCircle size={24} />
                    </div>
                    <h2 className="text-xl font-bold mb-2 text-red-700">Display Error</h2>
                    <p className="text-gray-600 mb-4">
                        This run could not be displayed correctly.
                    </p>
                    {this.props.runId && (
                        <div className="text-sm text-gray-500 mb-2">
                            Run ID: <code className="font-mono bg-gray-100 px-2 py-1 rounded">{this.props.runId}</code>
                        </div>
                    )}
                    {this.props.status && (
                        <div className="text-sm text-gray-500 mb-4">
                            Status: <span className="font-semibold">{this.props.status}</span>
                        </div>
                    )}
                    {this.state.error && (
                        <details className="text-left text-xs text-gray-400 bg-gray-50 p-3 rounded mt-4">
                            <summary className="cursor-pointer font-semibold">Technical Details</summary>
                            <pre className="mt-2 overflow-auto">{this.state.error.message}</pre>
                        </details>
                    )}
                    <button
                        onClick={this.props.onBack}
                        className="mt-8 px-4 py-2 border rounded hover:bg-gray-50"
                    >
                        Back to List
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}
