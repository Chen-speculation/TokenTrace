## ADDED Requirements

### Requirement: Logit Lens endpoint

The system SHALL provide `POST /api/logit-lens` that, given an input `context` and a resolved target token, returns the per-layer prediction trajectory: for each layer (including the embedding output as layer 0 and the final layer as layer L), the top-k candidate tokens and the target token's probability when that layer's final-position hidden state is projected through the model's final norm and lm_head. The target token selection SHALL match the existing prediction-attribution behavior: when neither `target_prediction` nor `target_token_id` is provided, the system MUST use the final-layer top-1 (greedy) next token; `target_prediction` (first token) and `target_token_id` MAY be provided but MUST be mutually exclusive.

#### Scenario: Default top-1 target

- **WHEN** a request provides `context`, `model`, and `source_page` but neither `target_prediction` nor `target_token_id`
- **THEN** the system resolves the target token via the final-layer greedy argmax and returns `success: true`, the resolved `target_token`, `n_layers`, `final_target_prob`, and a `layers` array of length `n_layers + 1`

#### Scenario: Explicit target token id

- **WHEN** a request provides a valid `target_token_id` within vocab range
- **THEN** the system projects every layer against that token id and reports its per-layer probability, including `final_target_prob` at the final layer

#### Scenario: Mutually exclusive targets rejected

- **WHEN** a request provides both `target_prediction` and `target_token_id`
- **THEN** the system responds with HTTP 400 and a message stating the two are mutually exclusive

#### Scenario: Missing or invalid input rejected

- **WHEN** `context` is missing or empty, or `model` is not one of `base`/`instruct`, or `source_page` is invalid
- **THEN** the system responds with HTTP 400 and a descriptive message

#### Scenario: Context exceeds length limit

- **WHEN** the tokenized `context` exceeds the attribution token-length limit
- **THEN** the system responds with an error indicating the limit and the current length, without running the model

### Requirement: Per-layer projection through final norm and lm_head

The system SHALL compute each layer's prediction by taking that layer's final-position hidden state from `forward(output_hidden_states=True)`, applying the model's final RMSNorm (`model.norm`) followed by the output embedding layer (`lm_head`), then softmax over the full vocabulary. The computation MUST NOT require backpropagation. The system MUST apply the norm in the same position the model itself applies it before lm_head, so that projecting the final layer reproduces the standard next-token logits.

#### Scenario: Final layer matches standard output

- **WHEN** the per-layer projection is applied to the final layer (layer L)
- **THEN** the resulting top-1 token MUST equal the argmax of `outputs.logits[:, -1, :]` (within `torch.isclose` tolerance on half-precision devices), serving as a correctness self-check

#### Scenario: Layer coverage includes embedding and final layer

- **WHEN** the response is returned
- **THEN** `layers` has length `n_layers + 1`, index 0 corresponds to the embedding output (marked `is_embedding: true`), and the last index corresponds to the final layer whose `target_prob` equals `final_target_prob`

#### Scenario: Single forward, no backward

- **WHEN** scoring all layers
- **THEN** the system runs exactly one forward pass with `output_hidden_states=True` and no backward pass

### Requirement: Memory-bounded layer projection

The system SHALL project layers one at a time and materialize only the top-k results plus the target token's probability per layer, discarding each layer's full-vocabulary logits tensor before processing the next. The system MUST NOT materialize an `[L, vocab_size]` tensor.

#### Scenario: Bounded peak memory

- **WHEN** the request is processed
- **THEN** at most one layer's `[1, vocab_size]` logits tensor is live at any time, and the returned `layers[].topk_tokens`/`topk_probs` are the only per-layer data retained

### Requirement: Attribution page layer trajectory panel

The Attribution page SHALL render a per-layer prediction trajectory panel: a layer-by-token heatmap of each layer's top-k candidates and a line chart of the target token's probability across layers. The panel SHALL highlight at least one summary landmark, such as the first layer at which the target token enters top-1.

#### Scenario: Trajectory panel shows layer formation

- **WHEN** the user runs a logit-lens request on the Attribution page for a given context and target
- **THEN** the page shows each layer's top-k candidates and the target token's probability trajectory across layers

#### Scenario: First-top-1 landmark reported

- **WHEN** the target token enters top-1 at some layer ℓ
- **THEN** the panel highlights ℓ as the "first top-1" layer; if it never enters top-1, the panel states that explicitly

#### Scenario: Interpretation disclaimer shown

- **WHEN** the trajectory panel is visible
- **THEN** a tooltip/disclaimer states that the values are projection readouts (final norm + lm_head applied to intermediate layers), not the model's literal per-layer computation
