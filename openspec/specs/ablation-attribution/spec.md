## ADDED Requirements

### Requirement: Ablation attribution endpoint

The system SHALL provide `POST /api/ablation-attribute` that, given an input `context` and a resolved target token, returns the causal effect of each input token on that target token's predicted probability. The target token selection SHALL match the existing prediction-attribution behavior: when neither `target_prediction` nor `target_token_id` is provided, the system MUST use the baseline top-1 (greedy) next token; `target_prediction` (first token) and `target_token_id` MAY be provided but MUST be mutually exclusive.

#### Scenario: Default top-1 target

- **WHEN** a request provides `context`, `model`, and `source_page` but neither `target_prediction` nor `target_token_id`
- **THEN** the system resolves the target token via baseline greedy argmax and returns `success: true`, the resolved `target_token`, the baseline `target_prob`, and a `token_attribution` array

#### Scenario: Explicit target token id

- **WHEN** a request provides a valid `target_token_id` within vocab range
- **THEN** the system attributes against that token id and returns its baseline `target_prob`

#### Scenario: Mutually exclusive targets rejected

- **WHEN** a request provides both `target_prediction` and `target_token_id`
- **THEN** the system responds with HTTP 400 and a message stating the two are mutually exclusive

#### Scenario: Missing or invalid input rejected

- **WHEN** `context` is missing or empty, or `model` is not one of `base`/`instruct`, or `source_page` is invalid
- **THEN** the system responds with HTTP 400 and a descriptive message

#### Scenario: Context exceeds length limit

- **WHEN** the tokenized `context` exceeds the attribution token-length limit
- **THEN** the system responds with an error indicating the limit and the current length, without running the model

### Requirement: Occlusion-based causal score

The system SHALL compute each token's score by occlusion: holding sequence length fixed, it replaces the token's input embedding with a neutral baseline vector (default: the mean of the context token embeddings), re-runs the forward pass, and sets `score = baseline_target_prob − occluded_target_prob`. The score MAY be negative (the token suppresses the target). The computation MUST NOT require backpropagation.

#### Scenario: Influential token yields measurable effect

- **WHEN** an input token strongly supports the target prediction
- **THEN** occluding it lowers the target probability, producing a positive `score`

#### Scenario: Baseline probability reported

- **WHEN** the request succeeds
- **THEN** the response includes the baseline `target_prob` used as the reference point for all token scores

#### Scenario: Single forward batch

- **WHEN** scoring N input tokens
- **THEN** the system evaluates the baseline and all N occluded variants in batched forward pass(es) without per-token backward passes, splitting into multiple batches only when memory limits require it

### Requirement: Coordinate alignment with gradient attribution

The returned `token_attribution` entries SHALL use the same character-offset scheme as the existing prediction-attribution response (each entry has `offset: [start, end]` into `context` and `raw`), excluding zero-width special tokens, so a client can overlay ablation scores on the same tokens as gradient attribution.

#### Scenario: Offsets map to context spans

- **WHEN** the response is returned
- **THEN** every `token_attribution` entry has `start < end`, `raw` equals `context[start:end]`, and special tokens with zero-width spans are omitted

### Requirement: Attribution page method comparison

The Attribution page SHALL let the user choose the attribution method among Gradient, Ablation, and Both. In Both mode the page SHALL render gradient and ablation scores over the same tokens with distinct visual encodings and display at least one consistency readout between the two methods.

#### Scenario: Both mode shows side-by-side comparison

- **WHEN** the user selects the Both method on the Attribution page for a given context and target
- **THEN** the page shows gradient-based and ablation-based highlights over the same tokens and a consistency metric (e.g. rank correlation or top-k overlap) between them

#### Scenario: Signed ablation encoding

- **WHEN** ablation scores include negative values
- **THEN** the page uses a diverging color scale distinguishing tokens that support the target from those that suppress it
