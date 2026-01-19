import { BrowserRouter, Routes, Route, Link, Navigate } from 'react-router-dom';
import RunsList from './components/RunsList';
import RunDetail from './components/RunDetail';
import { RunDetailErrorBoundary } from './components/RunDetailErrorBoundary';
import ResultsPage from './pages/ResultsPage';
import FindingsPage from './pages/FindingsPage';
import WorkbenchPage from './pages/WorkbenchPage';
import ExperimentPage from './pages/ExperimentPage';
import CandidatesPage from './pages/CandidatesPage';
import ExperimentsOpenPage from './pages/ExperimentsOpenPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
        {/* Navigation */}
        <nav style={{
          background: 'white',
          borderBottom: '1px solid #e5e7eb',
          padding: '1rem 2rem',
          display: 'flex',
          gap: '2rem',
          alignItems: 'center'
        }}>
          <Link to="/lab/runs" style={{ fontWeight: 'bold', fontSize: '1.2em', textDecoration: 'none', color: '#111827' }}>
            Model Lab
          </Link>
          <Link to="/lab/runs" style={{ textDecoration: 'none', color: '#4b5563' }}>Runs</Link>
          <Link to="/lab/workbench" style={{ textDecoration: 'none', color: '#4b5563' }}>Workbench</Link>
          <Link to="/lab/experiments" style={{ textDecoration: 'none', color: '#4b5563' }}>Experiments</Link>
          <Link to="/lab/candidates" style={{ textDecoration: 'none', color: '#4b5563' }}>Candidates</Link>
          <Link to="/lab/results" style={{ textDecoration: 'none', color: '#4b5563' }}>Results</Link>
          <Link to="/lab/findings" style={{ textDecoration: 'none', color: '#4b5563' }}>Findings</Link>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Navigate to="/lab/runs" replace />} />
          <Route path="/lab/runs" element={<RunsList onSelectRun={(id: string) => window.location.href = `/runs/${id}`} />} />
          <Route path="/runs/:runId" element={
            <RunDetailErrorBoundary onBack={() => window.location.href = '/lab/runs'}>
              <RunDetail onBack={() => window.location.href = '/lab/runs'} />
            </RunDetailErrorBoundary>
          } />

          <Route path="/lab/workbench" element={<WorkbenchPage />} />

          <Route path="/lab/experiments" element={<ExperimentsOpenPage />} />
          <Route path="/lab/experiments/:experimentId" element={<ExperimentPage />} />

          <Route path="/lab/candidates" element={<CandidatesPage />} />
          <Route path="/lab/results" element={<ResultsPage />} />
          <Route path="/lab/findings" element={<FindingsPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
