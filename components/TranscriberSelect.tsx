'use client';

import * as SelectPrimitive from '@radix-ui/react-select';
import { Check, ChevronDown, ChevronUp } from 'lucide-react';
import {
  type TranscriberId,
  type TranscriberOption,
} from '@/lib/transcribers';

type TranscriberSelectProps = {
  disabled: boolean;
  onValueChange: (value: TranscriberId) => void;
  options: readonly TranscriberOption[];
  value: TranscriberId;
};

export default function TranscriberSelect({
  disabled,
  onValueChange,
  options,
  value,
}: TranscriberSelectProps) {
  const selected = options.find((option) => option.value === value)
    ?? options[0];

  if (!selected) return null;

  return (
    <div className="transcriber-select-field">
      <span id="transcriber-select-label">TRANSCRIPTION ENGINE</span>
      <SelectPrimitive.Root
        value={value}
        onValueChange={(nextValue) => {
          onValueChange(nextValue as TranscriberId);
        }}
        disabled={disabled}
      >
        <SelectPrimitive.Trigger
          className="transcriber-select-trigger"
          aria-labelledby="transcriber-select-label transcriber-select-value"
        >
          <SelectPrimitive.Value>
            <span
              className="transcriber-select-value"
              id="transcriber-select-value"
            >
              <strong>{selected.label}</strong>
              {selected.detail ? <small>{selected.detail}</small> : null}
            </span>
          </SelectPrimitive.Value>
          <SelectPrimitive.Icon asChild>
            <ChevronDown aria-hidden="true" />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>

        <SelectPrimitive.Portal>
          <SelectPrimitive.Content
            className="transcriber-select-content"
            position="popper"
            side="top"
            sideOffset={8}
            align="end"
          >
            <SelectPrimitive.ScrollUpButton className="transcriber-select-scroll-button">
              <ChevronUp aria-hidden="true" />
            </SelectPrimitive.ScrollUpButton>
            <SelectPrimitive.Viewport className="transcriber-select-viewport">
              {options.map((option) => (
                <SelectPrimitive.Item
                  className="transcriber-select-item"
                  key={option.value}
                  value={option.value}
                >
                  <SelectPrimitive.ItemText>
                    <span className="transcriber-select-option">
                      <strong>{option.label}</strong>
                      {option.detail ? <small>{option.detail}</small> : null}
                    </span>
                  </SelectPrimitive.ItemText>
                  <SelectPrimitive.ItemIndicator className="transcriber-select-indicator">
                    <Check aria-hidden="true" />
                  </SelectPrimitive.ItemIndicator>
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.Viewport>
            <SelectPrimitive.ScrollDownButton className="transcriber-select-scroll-button">
              <ChevronDown aria-hidden="true" />
            </SelectPrimitive.ScrollDownButton>
          </SelectPrimitive.Content>
        </SelectPrimitive.Portal>
      </SelectPrimitive.Root>
    </div>
  );
}
