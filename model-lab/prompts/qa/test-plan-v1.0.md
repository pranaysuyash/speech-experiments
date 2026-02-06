# Test Plan Prompt

**Version**: 1.0  
**Purpose**: Create comprehensive test plans for features or changes.

## Use When

- Planning tests for a new feature
- Creating regression test suite
- Documenting manual test procedures

## Non-Negotiable Rules

1. **Coverage first**: Identify all test scenarios before writing tests
2. **Evidence required**: Each test must have pass/fail criteria
3. **Priority levels**: Not all tests are equal priority
4. **Reproducible**: Tests must be reproducible

## Inputs

- Feature or change description
- Acceptance criteria from ticket
- Existing test coverage

## Steps

### 1. Analyze Scope

Identify:
- Components affected
- Integration points
- User flows
- Edge cases

### 2. Test Categories

**Unit Tests**
- Individual function behavior
- Mock external dependencies
- Fast execution

**Integration Tests**
- Component interaction
- Database/file operations
- API endpoints

**End-to-End Tests**
- Full user flows
- Real dependencies
- Slowest but most realistic

**Model Tests** (for ASR/TTS)
- Inference correctness
- Performance benchmarks
- Memory usage
- Device compatibility

### 3. Test Case Template

```markdown
### TC-###: [Test Name]

**Category**: Unit/Integration/E2E/Model
**Priority**: P0/P1/P2
**Automated**: Yes/No

**Preconditions**:
- [setup required]

**Steps**:
1. [action]
2. [action]

**Expected Result**:
- [outcome]

**Pass Criteria**:
- [measurable criteria]
```

### 4. Priority Guidelines

**P0 (Critical)**
- Core functionality
- Security tests
- Data integrity

**P1 (High)**
- Common user flows
- Error handling
- Performance baselines

**P2 (Medium)**
- Edge cases
- Alternative flows
- UI polish

## Output Format

```markdown
# Test Plan: [Feature/Change Name]

**Ticket**: TCK-YYYYMMDD-###
**Author**: [name/agent]
**Date**: YYYY-MM-DD

## Scope

[What is being tested]

## Out of Scope

[What is NOT being tested]

## Test Environment

- Hardware: [MPS/CUDA/CPU]
- Python: [version]
- Dependencies: [key versions]

## Test Cases

### Unit Tests
[list test cases]

### Integration Tests
[list test cases]

### E2E Tests
[list test cases]

### Model Tests
[list test cases]

## Execution Plan

1. Run unit tests: `pytest tests/unit/`
2. Run integration tests: `pytest tests/integration/`
3. Run model benchmarks: `python -m harness.run --benchmark`

## Success Criteria

- [ ] All P0 tests pass
- [ ] All P1 tests pass
- [ ] P2 tests: [target]% pass rate
- [ ] No regressions in existing tests
```

## Stop Condition

Stop when:
- All test scenarios identified
- Test cases documented
- Priority assigned to each
- Execution plan defined
