import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import TranscriberSelect from '../components/TranscriberSelect';
import { TRANSCRIBERS } from '../lib/transcribers';

for (const option of TRANSCRIBERS) {
  const markup = renderToStaticMarkup(
    <TranscriberSelect
      disabled={false}
      onValueChange={() => {}}
      options={TRANSCRIBERS}
      value={option.value}
    />,
  );

  assert.match(markup, /role="combobox"/);
  assert.ok(markup.includes(option.label));
  if (option.detail) {
    assert.ok(markup.includes(option.detail));
  } else {
    assert.ok(!markup.includes('<small>'));
  }
}

const disabledMarkup = renderToStaticMarkup(
  <TranscriberSelect
    disabled
    onValueChange={() => {}}
    options={TRANSCRIBERS}
    value="gemini-transcribe"
  />,
);

assert.match(disabledMarkup, /disabled=""/);
console.log('Transcriber select rendering contract passed.');
