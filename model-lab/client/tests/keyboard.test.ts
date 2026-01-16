import { keyboardReducer, KeyboardAction, KeyboardState } from '../src/lib/keyboard';
import assert from 'assert';

console.log("Test 1: Next/Prev navigation");
let state: KeyboardState = { segmentsCount: 5, selectedIndex: null };

// Initial J -> 0
state = keyboardReducer(state, { type: 'NEXT' });
assert.strictEqual(state.selectedIndex, 0);

// Next -> 1
state = keyboardReducer(state, { type: 'NEXT' });
assert.strictEqual(state.selectedIndex, 1);

// K -> 0
state = keyboardReducer(state, { type: 'PREV' });
assert.strictEqual(state.selectedIndex, 0);

// K at 0 -> 0 (Clamp)
state = keyboardReducer(state, { type: 'PREV' });
assert.strictEqual(state.selectedIndex, 0);

// J to end
state = keyboardReducer(state, { type: 'NEXT' }); // 1
state = keyboardReducer(state, { type: 'NEXT' }); // 2
state = keyboardReducer(state, { type: 'NEXT' }); // 3
state = keyboardReducer(state, { type: 'NEXT' }); // 4
state = keyboardReducer(state, { type: 'NEXT' }); // 4 (Clamp)
assert.strictEqual(state.selectedIndex, 4);

console.log("Test 2: Empty list");
let emptyState: KeyboardState = { segmentsCount: 0, selectedIndex: null };
emptyState = keyboardReducer(emptyState, { type: 'NEXT' });
assert.strictEqual(emptyState.selectedIndex, null);

console.log("All tests passed!");
