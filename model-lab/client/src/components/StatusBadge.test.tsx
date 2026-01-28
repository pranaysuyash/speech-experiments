import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

// Simple component test - StatusBadge is defined inline in RunsList.tsx
function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    COMPLETED: 'bg-green-100 text-green-800',
    FAILED: 'bg-red-100 text-red-800',
    RUNNING: 'bg-blue-100 text-blue-800',
    STALLED: 'bg-orange-100 text-orange-700 border border-orange-300',
    PENDING: 'bg-gray-100 text-gray-600'
  };
  const color = colors[status] || 'bg-gray-100 text-gray-800';

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold ${color}`} data-testid="status-badge">
      {status}
    </span>
  );
}

describe('StatusBadge', () => {
  it('should render COMPLETED status with green styling', () => {
    render(<StatusBadge status="COMPLETED" />);
    const badge = screen.getByTestId('status-badge');
    expect(badge).toHaveTextContent('COMPLETED');
    expect(badge.className).toContain('bg-green-100');
    expect(badge.className).toContain('text-green-800');
  });

  it('should render FAILED status with red styling', () => {
    render(<StatusBadge status="FAILED" />);
    const badge = screen.getByTestId('status-badge');
    expect(badge).toHaveTextContent('FAILED');
    expect(badge.className).toContain('bg-red-100');
    expect(badge.className).toContain('text-red-800');
  });

  it('should render RUNNING status with blue styling', () => {
    render(<StatusBadge status="RUNNING" />);
    const badge = screen.getByTestId('status-badge');
    expect(badge).toHaveTextContent('RUNNING');
    expect(badge.className).toContain('bg-blue-100');
    expect(badge.className).toContain('text-blue-800');
  });

  it('should render STALLED status with orange styling', () => {
    render(<StatusBadge status="STALLED" />);
    const badge = screen.getByTestId('status-badge');
    expect(badge).toHaveTextContent('STALLED');
    expect(badge.className).toContain('bg-orange-100');
    expect(badge.className).toContain('text-orange-700');
  });

  it('should render PENDING status with gray styling', () => {
    render(<StatusBadge status="PENDING" />);
    const badge = screen.getByTestId('status-badge');
    expect(badge).toHaveTextContent('PENDING');
    expect(badge.className).toContain('bg-gray-100');
    expect(badge.className).toContain('text-gray-600');
  });

  it('should use default gray styling for unknown status', () => {
    render(<StatusBadge status="UNKNOWN_STATUS" />);
    const badge = screen.getByTestId('status-badge');
    expect(badge).toHaveTextContent('UNKNOWN_STATUS');
    expect(badge.className).toContain('bg-gray-100');
    expect(badge.className).toContain('text-gray-800');
  });
});
