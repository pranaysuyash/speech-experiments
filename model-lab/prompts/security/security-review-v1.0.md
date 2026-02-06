# Security Review Prompt

**Version**: 1.0  
**Purpose**: Review code for security vulnerabilities.

## Use When

- Before production deployment
- After adding new endpoints
- When handling user input or files
- Regular security audits

## Non-Negotiable Rules

1. **No secrets in code**: Ever
2. **Input validation**: All user input must be validated
3. **Least privilege**: Minimal permissions required
4. **Defense in depth**: Multiple layers of protection

## Security Checklist

### Authentication & Authorization

- [ ] Authentication required for protected endpoints
- [ ] Authorization checked for each operation
- [ ] Session management secure
- [ ] Password handling follows best practices

### Input Validation

- [ ] All user input validated
- [ ] File uploads validated (type, size, content)
- [ ] Path traversal prevented
- [ ] SQL injection prevented
- [ ] Command injection prevented

### Data Protection

- [ ] Sensitive data encrypted at rest
- [ ] Sensitive data encrypted in transit
- [ ] PII handling compliant
- [ ] Logging doesn't expose secrets

### API Security

- [ ] Rate limiting implemented
- [ ] CORS configured correctly
- [ ] Content-Type validated
- [ ] Response headers secure

### File Handling

- [ ] File paths validated
- [ ] Symlink attacks prevented
- [ ] Temporary files cleaned up
- [ ] Upload directory outside web root

### Dependencies

- [ ] Dependencies up to date
- [ ] No known vulnerabilities
- [ ] Minimal dependencies

## Common Vulnerabilities to Check

### Path Traversal

```python
# BAD
path = f"uploads/{user_input}"

# GOOD
from pathlib import Path
safe_path = Path("uploads").resolve() / Path(user_input).name
if not str(safe_path).startswith(str(Path("uploads").resolve())):
    raise ValueError("Invalid path")
```

### Command Injection

```python
# BAD
os.system(f"convert {user_file}")

# GOOD
import subprocess
subprocess.run(["convert", user_file], check=True)
```

### SQL Injection

```python
# BAD
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# GOOD
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

## Output Format

```markdown
# Security Review: [Scope]

**Date**: YYYY-MM-DD
**Reviewer**: [name/agent]
**Severity Summary**: [Critical: X, High: X, Medium: X, Low: X]

## Findings

### [CRITICAL/HIGH/MEDIUM/LOW]-001: [Title]

**File**: [path:line]
**Category**: [Input Validation/Auth/etc]
**Description**: [what's wrong]
**Impact**: [what could happen]
**Recommendation**: [how to fix]

**Code**:
```python
# Current (vulnerable)
...

# Recommended (secure)
...
```

## Recommendations

1. [Priority fixes]
2. [Medium-term improvements]
3. [Best practices to adopt]

## Verification

Commands to verify fixes:
```bash
[commands]
```
```

## Stop Condition

Stop when:
- All code in scope reviewed
- All findings documented
- Recommendations provided
- Severity levels assigned
