import { xToTime, timeToX, downsampleStride } from '../src/lib/timeline';
import assert from 'assert';

console.log("Test 1: xToTime");
assert.strictEqual(xToTime(0, 100, 60), 0);
assert.strictEqual(xToTime(50, 100, 60), 30);
assert.strictEqual(xToTime(100, 100, 60), 60);
// Clamp
assert.strictEqual(xToTime(150, 100, 60), 60);
assert.strictEqual(xToTime(-10, 100, 60), 0);

console.log("Test 2: timeToX");
assert.strictEqual(timeToX(0, 100, 60), 0);
assert.strictEqual(timeToX(30, 100, 60), 50);

console.log("Test 3: downsampleStride");
assert.strictEqual(downsampleStride(100, 2000), 1);
assert.strictEqual(downsampleStride(2000, 2000), 1);
assert.strictEqual(downsampleStride(2001, 2000), 2);
assert.strictEqual(downsampleStride(4000, 2000), 2);
assert.strictEqual(downsampleStride(4001, 2000), 3);

console.log("All tests passed!");
