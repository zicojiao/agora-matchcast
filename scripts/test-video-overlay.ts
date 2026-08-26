import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const component = readFileSync('components/MatchCastDemo.tsx', 'utf8');
const styles = readFileSync('app/globals.css', 'utf8');

assert.match(component, />\s*LIVE CAPTIONS\s*</);
assert.match(component, /className="live-pill"><i \/> LIVE<\/span>/);
assert.doesNotMatch(component, /AGORA LIVE(?: CAPTIONS)?/);

const captionRule = styles.match(/\.broadcast-caption p \{([\s\S]*?)\n\}/)?.[1];
assert.ok(captionRule, 'Broadcast caption CSS rule is missing.');
assert.match(captionRule, /font-size:\s*clamp\(16px, 1\.45vw, 20px\)/);
assert.match(captionRule, /-webkit-line-clamp:\s*3/);

console.log('Video overlay copy and typography contract passed.');
