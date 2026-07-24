package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/samber/lo"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

// --- helpers ---------------------------------------------------------------

func feedStream(t *testing.T, chunks []*model.InternalLLMResponse) []ResponsesStreamEvent {
	t.Helper()
	i := &ResponseInbound{}
	ctx := context.Background()

	var buf bytes.Buffer
	for _, c := range chunks {
		out, err := i.TransformStream(ctx, c)
		if err != nil {
			t.Fatalf("TransformStream failed: %v", err)
		}
		buf.Write(out)
	}

	return parseSSEEvents(t, buf.Bytes())
}

func parseSSEEvents(t *testing.T, raw []byte) []ResponsesStreamEvent {
	t.Helper()
	events := make([]ResponsesStreamEvent, 0)
	for _, line := range bytes.Split(raw, []byte("\n\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		payload := bytes.TrimPrefix(line, []byte("data: "))
		if bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var ev ResponsesStreamEvent
		if err := json.Unmarshal(payload, &ev); err != nil {
			t.Fatalf("failed to decode SSE event %q: %v", string(payload), err)
		}
		events = append(events, ev)
	}
	return events
}

func eventTypes(events []ResponsesStreamEvent) []string {
	out := make([]string, len(events))
	for i, e := range events {
		out[i] = e.Type
	}
	return out
}

func findEvent(events []ResponsesStreamEvent, t string) *ResponsesStreamEvent {
	for i := range events {
		if events[i].Type == t {
			return &events[i]
		}
	}
	return nil
}

func findItemDone(events []ResponsesStreamEvent, itemType string) *ResponsesItem {
	for i := range events {
		if events[i].Type != "response.output_item.done" {
			continue
		}
		if events[i].Item == nil {
			continue
		}
		if events[i].Item.Type == itemType {
			return events[i].Item
		}
	}
	return nil
}

func chunkWithDelta(model_ string, delta *model.Message) *model.InternalLLMResponse {
	return &model.InternalLLMResponse{
		ID:      "resp_test",
		Model:   model_,
		Object:  "chat.completion.chunk",
		Created: 123,
		Choices: []model.Choice{{Index: 0, Delta: delta}},
	}
}

func chunkWithFinish(model_, reason string) *model.InternalLLMResponse {
	r := reason
	return &model.InternalLLMResponse{
		ID:      "resp_test",
		Model:   model_,
		Object:  "chat.completion.chunk",
		Created: 123,
		Choices: []model.Choice{{Index: 0, Delta: &model.Message{}, FinishReason: &r}},
		Usage:   &model.Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
	}
}

// --- tests -----------------------------------------------------------------

func TestStreamReasoningBlocksSingleSignature(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("claude", &model.Message{
			ReasoningContent: lo.ToPtr("thinking..."),
		}),
		chunkWithDelta("claude", &model.Message{
			ReasoningBlocks: []model.ReasoningBlock{{
				Kind:      model.ReasoningBlockKindSignature,
				Signature: "sigA",
				Provider:  "anthropic",
			}},
		}),
		chunkWithFinish("claude", "stop"),
	}

	events := feedStream(t, chunks)

	item := findItemDone(events, "reasoning")
	if item == nil {
		t.Fatalf("reasoning item.done not found; got %v", eventTypes(events))
	}
	if item.EncryptedContent == nil || *item.EncryptedContent != "sigA" {
		t.Fatalf("expected encrypted_content=\"sigA\", got %v", item.EncryptedContent)
	}
	if findEvent(events, "response.reasoning.delta") != nil || findEvent(events, "response.reasoning.done") != nil {
		t.Fatalf("non-standard reasoning events must not be emitted, got %v", eventTypes(events))
	}
	if delta := findEvent(events, "response.reasoning_summary_text.delta"); delta == nil || delta.Delta != "thinking..." {
		t.Fatalf("expected reasoning summary delta, got %+v", delta)
	}
	if done := findEvent(events, "response.reasoning_summary_text.done"); done == nil || done.Text != "thinking..." {
		t.Fatalf("expected reasoning summary done with full text, got %+v", done)
	}
}

func TestStreamReasoningOutputItemAddedIncludesEmptySummary(t *testing.T) {
	i := &ResponseInbound{}
	out, err := i.TransformStreamEvents(context.Background(), []model.StreamEvent{
		{
			Kind:  model.StreamEventKindMessageStart,
			ID:    "resp_grok_cli",
			Model: "grok-4.5",
			Role:  "assistant",
		},
		{
			Kind:  model.StreamEventKindThinkingDelta,
			ID:    "resp_grok_cli",
			Model: "grok-4.5",
			Delta: &model.StreamDelta{Thinking: "The user is greeting"},
		},
	})
	if err != nil {
		t.Fatalf("TransformStreamEvents failed: %v", err)
	}

	for _, block := range bytes.Split(out, []byte("\n\n")) {
		payload := bytes.TrimPrefix(bytes.TrimSpace(block), []byte("data: "))
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var event map[string]json.RawMessage
		if err := json.Unmarshal(payload, &event); err != nil {
			t.Fatalf("unmarshal SSE event: %v", err)
		}
		var eventType string
		if err := json.Unmarshal(event["type"], &eventType); err != nil || eventType != "response.output_item.added" {
			continue
		}

		var item map[string]json.RawMessage
		if err := json.Unmarshal(event["item"], &item); err != nil {
			t.Fatalf("unmarshal output item: %v", err)
		}
		summary, ok := item["summary"]
		if !ok {
			t.Fatalf("reasoning output_item.added omitted required summary: %s", payload)
		}
		if !bytes.Equal(bytes.TrimSpace(summary), []byte("[]")) {
			t.Fatalf("reasoning output_item.added summary = %s, want []", summary)
		}
		return
	}

	t.Fatal("reasoning response.output_item.added event not found")
}

func TestStreamReasoningSummaryPartAddedIncludesEmptyText(t *testing.T) {
	i := &ResponseInbound{}
	out, err := i.TransformStreamEvents(context.Background(), []model.StreamEvent{
		{
			Kind:  model.StreamEventKindMessageStart,
			ID:    "resp_grok_cli",
			Model: "grok-4.5",
			Role:  "assistant",
		},
		{
			Kind:  model.StreamEventKindThinkingDelta,
			ID:    "resp_grok_cli",
			Model: "grok-4.5",
			Delta: &model.StreamDelta{Thinking: "The user is greeting"},
		},
	})
	if err != nil {
		t.Fatalf("TransformStreamEvents failed: %v", err)
	}

	for _, block := range bytes.Split(out, []byte("\n\n")) {
		payload := bytes.TrimPrefix(bytes.TrimSpace(block), []byte("data: "))
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var event map[string]json.RawMessage
		if err := json.Unmarshal(payload, &event); err != nil {
			t.Fatalf("unmarshal SSE event: %v", err)
		}
		var eventType string
		if err := json.Unmarshal(event["type"], &eventType); err != nil || eventType != "response.reasoning_summary_part.added" {
			continue
		}

		var part map[string]json.RawMessage
		if err := json.Unmarshal(event["part"], &part); err != nil {
			t.Fatalf("unmarshal reasoning summary part: %v", err)
		}
		text, ok := part["text"]
		if !ok {
			t.Fatalf("reasoning_summary_part.added omitted required text: %s", payload)
		}
		if !bytes.Equal(bytes.TrimSpace(text), []byte(`""`)) {
			t.Fatalf("reasoning_summary_part.added text = %s, want empty string", text)
		}
		return
	}

	t.Fatal("response.reasoning_summary_part.added event not found")
}

func TestStreamOutputTextIncludesEmptyAnnotations(t *testing.T) {
	events := feedStream(t, []*model.InternalLLMResponse{
		chunkWithDelta("grok-4.5", &model.Message{
			Content: model.MessageContent{Content: lo.ToPtr("你好")},
		}),
		chunkWithFinish("grok-4.5", "stop"),
	})

	var addedPart, donePart *ResponsesContentPart
	for idx := range events {
		if events[idx].Type == "response.content_part.added" && events[idx].Part != nil && events[idx].Part.Type == "output_text" {
			addedPart = events[idx].Part
		}
		if events[idx].Type == "response.content_part.done" && events[idx].Part != nil && events[idx].Part.Type == "output_text" {
			donePart = events[idx].Part
		}
	}
	for eventName, part := range map[string]*ResponsesContentPart{
		"response.content_part.added": addedPart,
		"response.content_part.done":  donePart,
	} {
		if part == nil || part.Annotations == nil || len(*part.Annotations) != 0 {
			t.Fatalf("%s must include annotations: [], got %+v", eventName, part)
		}
	}

	item := findItemDone(events, "message")
	if item == nil || item.Content == nil || len(item.Content.Items) != 1 {
		t.Fatalf("message output_item.done missing output_text content: %+v", item)
	}
	textItem := item.Content.Items[0]
	if textItem.Type != "output_text" || textItem.Annotations == nil || len(*textItem.Annotations) != 0 {
		t.Fatalf("output_item.done output_text must include annotations: [], got %+v", textItem)
	}
}

func TestStreamReasoningBlocksMultipleSignatures(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("claude", &model.Message{
			ReasoningContent: lo.ToPtr("block one"),
		}),
		chunkWithDelta("claude", &model.Message{
			ReasoningBlocks: []model.ReasoningBlock{{
				Kind: model.ReasoningBlockKindSignature, Signature: "sig1", Provider: "anthropic",
			}},
		}),
		chunkWithDelta("claude", &model.Message{
			ReasoningContent: lo.ToPtr("block two"),
		}),
		chunkWithDelta("claude", &model.Message{
			ReasoningBlocks: []model.ReasoningBlock{{
				Kind: model.ReasoningBlockKindSignature, Signature: "sig2", Provider: "anthropic",
			}},
		}),
		chunkWithFinish("claude", "stop"),
	}

	events := feedStream(t, chunks)

	item := findItemDone(events, "reasoning")
	if item == nil || item.EncryptedContent == nil {
		t.Fatalf("reasoning item with encrypted_content missing")
	}
	var decoded []string
	if err := json.Unmarshal([]byte(*item.EncryptedContent), &decoded); err != nil {
		t.Fatalf("multi-sig encrypted_content should be JSON array, got %q: %v", *item.EncryptedContent, err)
	}
	if len(decoded) != 2 || decoded[0] != "sig1" || decoded[1] != "sig2" {
		t.Fatalf("unexpected signatures: %v", decoded)
	}
}

func TestStreamReasoningBlocksRedacted(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("claude", &model.Message{
			ReasoningBlocks: []model.ReasoningBlock{{
				Kind:     model.ReasoningBlockKindRedacted,
				Data:     "REDACTED_DATA",
				Provider: "anthropic",
			}},
		}),
		chunkWithFinish("claude", "stop"),
	}

	events := feedStream(t, chunks)

	if findEvent(events, "response.output_item.added") == nil {
		t.Fatalf("redacted block should open a reasoning item; events=%v", eventTypes(events))
	}
	if findItemDone(events, "reasoning") == nil {
		t.Fatalf("redacted block should close with reasoning output_item.done; events=%v", eventTypes(events))
	}
}

func TestStreamReasoningLegacyFallback(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("openrouter", &model.Message{
			ReasoningContent: lo.ToPtr("legacy reasoning"),
		}),
		chunkWithDelta("openrouter", &model.Message{
			ReasoningSignature: lo.ToPtr("sigLegacy"),
		}),
		chunkWithFinish("openrouter", "stop"),
	}

	events := feedStream(t, chunks)

	item := findItemDone(events, "reasoning")
	if item == nil || item.EncryptedContent == nil {
		t.Fatalf("legacy reasoning path lost signature; events=%v", eventTypes(events))
	}
	if *item.EncryptedContent != "sigLegacy" {
		t.Fatalf("expected legacy signature verbatim, got %q", *item.EncryptedContent)
	}
}

func TestStreamRefusalEvents(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("claude", &model.Message{Refusal: "I cannot"}),
		chunkWithDelta("claude", &model.Message{Refusal: " help with that."}),
		chunkWithFinish("claude", "refusal"),
	}

	events := feedStream(t, chunks)
	types := eventTypes(events)

	want := []string{
		"response.content_part.added",
		"response.refusal.delta",
		"response.refusal.delta",
	}
	for _, w := range want {
		if findEvent(events, w) == nil {
			t.Fatalf("expected event %s, got sequence %v", w, types)
		}
	}

	addPart := findEvent(events, "response.content_part.added")
	if addPart.Part == nil || addPart.Part.Type != "refusal" {
		t.Fatalf("content_part.added should be refusal, got %+v", addPart.Part)
	}

	refusalDone := findEvent(events, "response.refusal.done")
	if refusalDone == nil || refusalDone.Text != "I cannot help with that." {
		t.Fatalf("refusal.done text mismatch: got %v", refusalDone)
	}

	partDone := findEvent(events, "response.content_part.done")
	if partDone == nil || partDone.Part == nil || partDone.Part.Type != "refusal" {
		t.Fatalf("content_part.done should carry Type=refusal, got %+v", partDone)
	}

	item := findItemDone(events, "message")
	if item == nil || item.Content == nil || len(item.Content.Items) == 0 {
		t.Fatalf("message item.done missing; events=%v", types)
	}
	if item.Content.Items[0].Refusal == nil || *item.Content.Items[0].Refusal != "I cannot help with that." {
		t.Fatalf("item refusal content lost: %+v", item.Content.Items[0].Refusal)
	}
}

func TestStreamToolCallArgumentDeltaUsesStoredOutputIndex(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("gpt-4o", &model.Message{
			ToolCalls: []model.ToolCall{{
				Index: 0,
				ID:    "call_a",
				Type:  "function",
				Function: model.FunctionCall{
					Name:      "first",
					Arguments: `{"a":`,
				},
			}},
		}),
		chunkWithDelta("gpt-4o", &model.Message{
			ToolCalls: []model.ToolCall{{
				Index: 1,
				ID:    "call_b",
				Type:  "function",
				Function: model.FunctionCall{
					Name:      "second",
					Arguments: `{"b":1}`,
				},
			}},
		}),
		chunkWithDelta("gpt-4o", &model.Message{
			ToolCalls: []model.ToolCall{{
				Index: 0,
				ID:    "call_a",
				Type:  "function",
				Function: model.FunctionCall{
					Arguments: `1}`,
				},
			}},
		}),
	}

	events := feedStream(t, chunks)
	var firstToolOutputIndex *int
	for _, event := range events {
		if event.Type == "response.output_item.added" && event.Item != nil && event.Item.CallID == "call_a" {
			firstToolOutputIndex = event.OutputIndex
			break
		}
	}
	if firstToolOutputIndex == nil {
		t.Fatalf("expected output_item.added for call_a; events=%v", eventTypes(events))
	}

	var lateDelta *ResponsesStreamEvent
	for i := range events {
		if events[i].Type == "response.function_call_arguments.delta" && events[i].ItemID != nil && *events[i].ItemID == "call_a" && events[i].Delta == `1}` {
			lateDelta = &events[i]
			break
		}
	}
	if lateDelta == nil {
		t.Fatalf("expected late argument delta for call_a; events=%v", eventTypes(events))
	}
	if lateDelta.OutputIndex == nil || *lateDelta.OutputIndex != *firstToolOutputIndex {
		t.Fatalf("late call_a delta output_index=%v, want %d", lateDelta.OutputIndex, *firstToolOutputIndex)
	}
}

func TestStreamFunctionCallAddedIncludesEmptyArguments(t *testing.T) {
	i := &ResponseInbound{}
	out, err := i.TransformStream(context.Background(), chunkWithDelta("grok-4.5", &model.Message{
		ToolCalls: []model.ToolCall{{
			Index: 0,
			ID:    "call_weather",
			Type:  "function",
			Function: model.FunctionCall{
				Name: "get_weather",
			},
		}},
	}))
	if err != nil {
		t.Fatalf("TransformStream failed: %v", err)
	}

	for _, block := range bytes.Split(out, []byte("\n\n")) {
		payload := bytes.TrimPrefix(bytes.TrimSpace(block), []byte("data: "))
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var event map[string]json.RawMessage
		if err := json.Unmarshal(payload, &event); err != nil {
			t.Fatalf("unmarshal SSE event: %v", err)
		}
		var eventType string
		if err := json.Unmarshal(event["type"], &eventType); err != nil || eventType != "response.output_item.added" {
			continue
		}

		var item map[string]json.RawMessage
		if err := json.Unmarshal(event["item"], &item); err != nil {
			t.Fatalf("unmarshal function call item: %v", err)
		}
		arguments, ok := item["arguments"]
		if !ok {
			t.Fatalf("function_call output_item.added omitted required arguments: %s", payload)
		}
		if !bytes.Equal(bytes.TrimSpace(arguments), []byte(`""`)) {
			t.Fatalf("function_call arguments = %s, want empty string", arguments)
		}
		if _, ok := item["call_id"]; !ok {
			t.Fatalf("function_call output_item.added omitted required call_id: %s", payload)
		}
		return
	}

	t.Fatal("function_call response.output_item.added event not found")
}

func TestTransformStreamEventsSignatureOnlyStillOpensReasoningItem(t *testing.T) {
	i := &ResponseInbound{}
	ctx := context.Background()

	out, err := i.TransformStreamEvents(ctx, []model.StreamEvent{
		{
			Kind:  model.StreamEventKindMessageStart,
			ID:    "resp_sig_only",
			Model: "claude",
			Role:  "assistant",
		},
		{
			Kind:  model.StreamEventKindSignatureDelta,
			ID:    "resp_sig_only",
			Model: "claude",
			Delta: &model.StreamDelta{Signature: "sig_only"},
		},
		{
			Kind:       model.StreamEventKindMessageStop,
			ID:         "resp_sig_only",
			Model:      "claude",
			StopReason: model.FinishReasonStop,
		},
		{
			Kind:  model.StreamEventKindUsageDelta,
			ID:    "resp_sig_only",
			Model: "claude",
			Usage: &model.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		},
	})
	if err != nil {
		t.Fatalf("TransformStreamEvents failed: %v", err)
	}

	events := parseSSEEvents(t, out)
	item := findItemDone(events, "reasoning")
	if item == nil || item.EncryptedContent == nil || *item.EncryptedContent != "sig_only" {
		t.Fatalf("expected reasoning output item with encrypted_content, got %+v", item)
	}
}

func TestTransformStreamMatchesStreamEventsProjection(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("claude", &model.Message{
			Role:             "assistant",
			ReasoningContent: lo.ToPtr("thinking..."),
		}),
		chunkWithDelta("claude", &model.Message{
			ReasoningBlocks: []model.ReasoningBlock{{
				Kind:      model.ReasoningBlockKindSignature,
				Signature: "sigA",
				Provider:  "anthropic",
			}},
		}),
		chunkWithDelta("claude", &model.Message{
			Content: model.MessageContent{Content: lo.ToPtr("answer")},
			Refusal: " but not that part",
		}),
		chunkWithFinish("claude", "stop"),
	}

	streamInbound := &ResponseInbound{}
	eventInbound := &ResponseInbound{}
	ctx := context.Background()

	var streamBuf bytes.Buffer
	var eventBuf bytes.Buffer
	for _, chunk := range chunks {
		streamOut, err := streamInbound.TransformStream(ctx, chunk)
		if err != nil {
			t.Fatalf("TransformStream failed: %v", err)
		}
		streamBuf.Write(streamOut)

		eventOut, err := eventInbound.TransformStreamEvents(ctx, model.StreamEventsFromInternalResponse(chunk))
		if err != nil {
			t.Fatalf("TransformStreamEvents failed: %v", err)
		}
		eventBuf.Write(eventOut)
	}

	streamEvents := parseSSEEvents(t, streamBuf.Bytes())
	projectedEvents := parseSSEEvents(t, eventBuf.Bytes())
	if len(streamEvents) != len(projectedEvents) {
		t.Fatalf("event count mismatch: stream=%d projected=%d", len(streamEvents), len(projectedEvents))
	}
	for idx := range streamEvents {
		if streamEvents[idx].Type != projectedEvents[idx].Type {
			t.Fatalf("event[%d] type mismatch: %q vs %q", idx, streamEvents[idx].Type, projectedEvents[idx].Type)
		}
	}

	streamDone := findItemDone(streamEvents, "message")
	projectedDone := findItemDone(projectedEvents, "message")
	if streamDone == nil || projectedDone == nil {
		t.Fatalf("expected message output_item.done in both paths")
	}
	streamDone.ID = ""
	projectedDone.ID = ""
	gotStream, err := json.Marshal(streamDone)
	if err != nil {
		t.Fatalf("marshal stream item: %v", err)
	}
	gotProjected, err := json.Marshal(projectedDone)
	if err != nil {
		t.Fatalf("marshal projected item: %v", err)
	}
	if string(gotStream) != string(gotProjected) {
		t.Fatalf("message item mismatch:\nstream=%s\nprojected=%s", gotStream, gotProjected)
	}
}

func TestStreamTextThenRefusal(t *testing.T) {
	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("claude", &model.Message{
			Content: model.MessageContent{Content: lo.ToPtr("partial answer...")},
		}),
		chunkWithDelta("claude", &model.Message{Refusal: "Actually I can't."}),
		chunkWithFinish("claude", "refusal"),
	}

	events := feedStream(t, chunks)
	types := eventTypes(events)

	// Expect text part fully opened-and-closed before refusal opens.
	textDoneIdx := -1
	refusalAddedIdx := -1
	for idx, e := range events {
		if e.Type == "response.output_text.done" && textDoneIdx == -1 {
			textDoneIdx = idx
		}
		if e.Type == "response.content_part.added" && e.Part != nil && e.Part.Type == "refusal" && refusalAddedIdx == -1 {
			refusalAddedIdx = idx
		}
	}
	if textDoneIdx == -1 || refusalAddedIdx == -1 {
		t.Fatalf("expected text.done then refusal part.added; got %v", types)
	}
	if textDoneIdx > refusalAddedIdx {
		t.Fatalf("text.done (%d) must precede refusal content_part.added (%d)", textDoneIdx, refusalAddedIdx)
	}

	item := findItemDone(events, "message")
	if item == nil || item.Content == nil {
		t.Fatalf("message item.done missing")
	}
	if len(item.Content.Items) != 2 {
		t.Fatalf("expected 2 content items (text + refusal), got %d: %+v", len(item.Content.Items), item.Content.Items)
	}
	if item.Content.Items[0].Type != "output_text" || item.Content.Items[1].Type != "refusal" {
		t.Fatalf("content items order wrong: %+v", item.Content.Items)
	}
	if item.Content.Items[0].Text == nil || !strings.Contains(*item.Content.Items[0].Text, "partial answer") {
		t.Fatalf("text content lost: %+v", item.Content.Items[0].Text)
	}
	if item.Content.Items[1].Refusal == nil || *item.Content.Items[1].Refusal != "Actually I can't." {
		t.Fatalf("refusal content lost: %+v", item.Content.Items[1].Refusal)
	}
}
