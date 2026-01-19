# Kill Switch Procedures

**Question**: If usage spikes or something goes wrong, how do we stop damage in < 5 minutes?

## Actions

1. **Hard Stop (Backend)**:
   `kill -9 $(lsof -ti:8000)`

2. **Prevent New Runs (Config)**:
   Set environment variable `MAX_CONCURRENT_RUNS=0` and restart server.

3. **Global Lockout (Frontend)**:
   `kill -9 $(lsof -ti:5174)`
