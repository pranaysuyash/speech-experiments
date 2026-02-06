# Code Review Checklist

**Version**: 1.0  
**Purpose**: Quick checklist for reviewing code changes.

## Before Starting

- [ ] Read the related ticket/PR description
- [ ] Understand the scope and acceptance criteria
- [ ] Check if this is a bug fix, feature, or refactor

## Correctness

- [ ] Does the code do what it's supposed to do?
- [ ] Are edge cases handled?
- [ ] Is error handling appropriate?
- [ ] Are there any obvious bugs?

## Code Quality

- [ ] Is the code readable and maintainable?
- [ ] Are variable/function names descriptive?
- [ ] Is there unnecessary duplication?
- [ ] Are comments helpful (not obvious)?
- [ ] Is complexity reasonable?

## Architecture

- [ ] Does it fit the existing patterns?
- [ ] Is it in the right layer (API/Service/Repo)?
- [ ] Are dependencies appropriate?
- [ ] Is it testable?

## Security

- [ ] No secrets or credentials in code?
- [ ] Input validation present?
- [ ] No SQL injection or command injection?
- [ ] Proper authentication/authorization?

## Performance

- [ ] No obvious performance issues?
- [ ] Appropriate use of async/await?
- [ ] No unnecessary loops or queries?
- [ ] Memory usage reasonable?

## Testing

- [ ] Tests cover the changes?
- [ ] Tests cover edge cases?
- [ ] Tests are readable and maintainable?
- [ ] No flaky test patterns?

## Documentation

- [ ] API changes documented?
- [ ] README updated if needed?
- [ ] Worklog ticket has evidence?

## Model-Specific (if applicable)

- [ ] Model registry updated?
- [ ] Benchmark results documented?
- [ ] Hardware requirements noted?
- [ ] Memory usage tested?

## Final Checks

- [ ] All tests pass?
- [ ] Build succeeds?
- [ ] No new warnings?
- [ ] Scope matches ticket?

## Verdict

- [ ] **APPROVE**: Ready to merge
- [ ] **REQUEST_CHANGES**: Issues must be fixed
- [ ] **COMMENT**: Questions or suggestions, not blocking
