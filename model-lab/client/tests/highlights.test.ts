import { highlightsApi, HighlightStore } from '../src/lib/highlights';
import assert from 'assert';

// Mock LocalStorage
const store: Record<string, string> = {};
global.localStorage = {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => {},
    length: 0,
    key: (index: number) => null
};

// Test 1: Get empty store
console.log("Test 1: Get empty store");
const s1 = highlightsApi.get("run_123");
assert.strictEqual(s1.run_id, "run_123");
assert.strictEqual(s1.items.length, 0);

// Test 2: Add highlight
console.log("Test 2: Add highlight");
const seg = { start_s: 10.5, end_s: 15.2, text: "Important stuff" };
highlightsApi.add("run_123", seg, "Note 1");

const s2 = highlightsApi.get("run_123");
assert.strictEqual(s2.items.length, 1);
assert.strictEqual(s2.items[0].text, "Important stuff");
assert.strictEqual(s2.items[0].note, "Note 1");
assert.strictEqual(s2.items[0].start_s, 10.5);

// Test 3: Persistence
console.log("Test 3: Persistence");
assert.ok(store['highlights:v1:run_123']);
const raw = JSON.parse(store['highlights:v1:run_123']);
assert.strictEqual(raw.run_id, "run_123");
assert.strictEqual(raw.items.length, 1);

// Test 4: Export Markdown
console.log("Test 4: Export Markdown");
const md = highlightsApi.exportMarkdown(s2, "meeting.wav");
console.log("Markdown Output:\n" + md);

assert.ok(md.includes("# Highlights — meeting.wav"));
assert.ok(md.includes("Run: run_123"));
assert.ok(md.includes("- [00:10.50 → 00:15.20] Important stuff"));
assert.ok(md.includes("  - Note: Note 1"));

// Test 5: Run Isolation
console.log("Test 5: Run Isolation");
const seg2 = { start_s: 0, end_s: 1, text: "Run B Start" };
highlightsApi.add("run_456", seg2, "Note Run B");

const storeA = highlightsApi.get("run_123");
const storeB = highlightsApi.get("run_456");

assert.strictEqual(storeA.items.length, 1);
assert.strictEqual(storeB.items.length, 1);
assert.strictEqual(storeA.items[0].text, "Important stuff");
assert.strictEqual(storeB.items[0].text, "Run B Start");

console.log("All tests passed!");

