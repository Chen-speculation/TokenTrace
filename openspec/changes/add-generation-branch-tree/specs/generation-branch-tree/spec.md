## ADDED Requirements

### Requirement: Branch-next endpoint

The system SHALL provide `POST /api/branch-next` that, given a determined `prefix` and `model`, returns the top-k candidate next tokens with their probabilities by running a single forward pass over the prefix and reading the final-position logits. The endpoint MUST NOT invoke the generation loop (`model.generate`) and MUST NOT require backpropagation.

#### Scenario: Default top-k expansion

- **WHEN** a request provides `prefix`, `model`, and `source_page` but no `top_k`
- **THEN** the system returns `success: true`, the default top-k candidates (each with `token`, `token_id`, `prob`), `prefix_tokens`, and `is_context_full: false`

#### Scenario: Custom top-k within limit

- **WHEN** a request provides a `top_k` between 1 and the configured upper bound
- **THEN** the system returns exactly that many candidates, ordered by descending probability

#### Scenario: top_k above upper bound clamped

- **WHEN** a request provides a `top_k` greater than the configured upper bound
- **THEN** the system clamps it to the upper bound and returns that many candidates

#### Scenario: Missing or invalid input rejected

- **WHEN** `prefix` is missing or empty, or `model` is not one of `base`/`instruct`, or `source_page` is invalid
- **THEN** the system responds with HTTP 400 and a descriptive message

### Requirement: Single forward pass without generation loop

The system SHALL compute the next-token candidates by encoding the `prefix`, running one forward pass with `use_cache=False` and `output_attentions=False`, and applying softmax + top-k to the final-position logits. The system MUST NOT start an autoregressive generation loop.

#### Scenario: One forward pass per request

- **WHEN** the request is processed
- **THEN** the model forward is invoked exactly once and no `model.generate` call occurs

#### Scenario: Candidates match final-position distribution

- **WHEN** the response is returned
- **THEN** the top-1 candidate's `token_id` equals the argmax of the final-position logits, and probabilities sum to the softmax of the returned top-k (within half-precision tolerance on MPS/CUDA)

### Requirement: Context-length guard on prefix

The system SHALL reject a `prefix` whose tokenized length leaves no room for a next position, reusing the existing `PromptTooLongError` semantics. When the prefix occupies the full context window, the system SHALL report `is_context_full: true` in any successful response that precedes the limit, and SHALL return an error when expansion is no longer possible.

#### Scenario: Prefix exceeds context limit

- **WHEN** the tokenized `prefix` reaches the model context limit
- **THEN** the system responds with an error indicating the prefix is too long and no further expansion is possible

#### Scenario: Context full flag reported before limit

- **WHEN** the prefix is within the limit but expanding one more position would reach it
- **THEN** the successful response includes `is_context_full: true` so the client can warn before the next expansion

### Requirement: Causal Flow page branch-tree mode

The Causal Flow page SHALL offer a "branch-tree" mode in which the user can expand any token node's top-k candidates and spawn a new branch by selecting a candidate, forming a tree of alternative generation futures. The mode SHALL enforce hard limits on tree depth, branching width, and total node count, and SHALL refuse expansion beyond those limits with a clear message.

#### Scenario: Expand root into candidates

- **WHEN** the user enters branch-tree mode and submits an initial prefix
- **THEN** the page renders a root node and its top-k candidates as selectable leaves

#### Scenario: Spawn a new branch by selecting a candidate

- **WHEN** the user selects a candidate leaf
- **THEN** the page forms a new prefix by appending the candidate's token text and expands the new branch node, maintaining a strict tree structure where each node has exactly one parent

#### Scenario: Expansion limits enforced

- **WHEN** the tree reaches the configured depth limit, branching-width limit, or total node-count limit
- **THEN** the page refuses further expansion and shows a message explaining which limit was hit

#### Scenario: Request cancellation and caching

- **WHEN** the user triggers a new expansion while a previous one is in flight, or re-expands an already-cached node
- **THEN** the page cancels the in-flight request via AbortController and serves the cached result for the re-expanded node without a new request
