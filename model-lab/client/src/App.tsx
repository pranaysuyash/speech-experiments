import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import RunsList from './components/RunsList';
import RunDetail from './components/RunDetail';
import ResultsPage from './pages/ResultsPage';
import FindingsPage from './pages/FindingsPage';
import WorkbenchPage from './pages/WorkbenchPage';

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
          <Link to="/" style={{ fontWeight: 'bold', fontSize: '1.2em', textDecoration: 'none', color: '#111827' }}>
            Model Lab
          </Link>
          <Link to="/" style={{ textDecoration: 'none', color: '#4b5563' }}>Runs</Link>
          <Link to="/lab/workbench" style={{ textDecoration: 'none', color: '#4b5563' }}>Workbench</Link>
          <Link to="/lab/results" style={{ textDecoration: 'none', color: '#4b5563' }}>Results</Link>
          <Link to="/lab/findings" style={{ textDecoration: 'none', color: '#4b5563' }}>Findings</Link>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<RunsList onSelectRun={(id: string) => window.location.href = `/runs/${id}`} />} />
          <Route path="/runs/:runId" element={<RunDetail onBack={() => window.location.href = '/'} />} />
          <Route path="/lab/workbench" element={<WorkbenchPage />} />
          <Route path="/lab/results" element={<ResultsPage />} />
          <Route path="/lab/findings" element={<FindingsPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
