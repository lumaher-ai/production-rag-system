## Day 2 — [04/13/2026]

**Done:**
- async/await fundamentals + asyncio.gather
- httpx async client with JSONPlaceholder
- First FastAPI app with /health and /echo endpoints validated with Pydantic
- pytest setup + 3 tests covering happy and sad paths
- Branch workflow adopted + PR

**Learned:**
- asyncio.gather is Promise.all
- sequential vs concurrent makes a 5x differece visibly
- FastAPI generates Swagger UI from Pydantic models automatically
- Defining the Pydantic model once generates validation, documentation, serialization, and type safety in the code that consumes the model. This is dramatically more productive than the Express + Joi + swagger-jsdoc approach.
- FastAPI's TestClient lets you make HTTP requests to your app without starting a real server. It's synchronous internally withou matter if my app is async — FastAPI handles the translation. Also it's incredibly fast.
- pytest assert-based tests have introspections which makes errors clearer


## Things I've seen but don't deeply understand yet (and that's OK)
- typing.Optional / typing.Union — know they're for "X or None", will learn more when I see them in real code
- typing.Generic, TypeVar — saw them mentioned, will revisit when I write something that needs them
- async/await internals — using them at surface level, will go deep when I hit a real problem
- __name__ == "__main__" - this is something important when using sample data and other files that import content must ignore the case use with data and only access to the abstraction, I will be more relevant when creating tests, but I do not how this is used when importing files and why this works
- Some modern conventions like ":.2f" used in the time.perf_counter() - start
- Why the design pattern decorater is extensively used in Pyton? ( decorators to define endpoints,  decorator @router.post(), @app.exception_handler(MyCustomError), lru_cache)

## Day 3 — [04/14/2026]

**Done:**
- Refactored FastAPI app into routes/schemas/services layers
- Added Pydantic Settings with .env file support
- Implemented dependency injection pattern for config
- Set up structlog with dev (colored) and prod (JSON) modes
- Added Field validations (min/max length, value ranges)
- Added tests for validation edge cases

**Learned:**
- Layered architecture separates "what data looks like" (schemas), "what the system does" (services), and "how HTTP maps to it" (routes)
- Pydantic Settings applies the same "validated typed data" pattern to configuration as Pydantic does to API data
- `@lru_cache` is the standard way to make settings behave as a singleton
- Structured logs are objects with fields, not strings — they make filtering in production observavility tools possible

**Things I've seen but don't deeply understand yet (and that's OK):**
- FastAPI's dependency injection internals (just using it at surface level)
- structlog's processor chain (trusting the setup, will revisit when I hit a problem)
- the asynccontextmanager decorator in the lifespan, and the yield keword(works for now, will learn more when I need custom lifecycle)

**Tomorrow:**
- Day 4: preparation for week 2 — Postgres with Docker, SQLAlchemy 2.0 async, Alembic migrations

## Day 4 — [04/15/2026]

**Done:**
- Postgres 16 running via Docker Compose with healthcheck and persistent volume
- asyncpg connection tested with raw SQL
- SQLAlchemy 2.0 async setup: engine, session factory, declarative Base
- First model: User with UUID PK, unique email
- Alembic initialized with async template; first migration generated and applied
- Repository pattern with UserRepository encapsulating DB access
- UserService wrapping the repository
- POST /users endpoint with full DI chain: route → service → repository → session
- Tests with SQLite in-memory and dependency_overrides
- 8 tests total passing (5 from before + 3 new)

**Learned:**
- Docker Compose isolates Postgres from my system; tear-down is `docker compose down -v`
- SQLAlchemy has 2 layers: Core (SQL builder) and ORM (class-to-table mapping); I'm using ORM
- `Mapped[type]` + `mapped_column(...)` is the modern 2.0 syntax; replaces old `Column(...)` style
- Alembic doesn't detect models unless you explicitly import them in env.py
- `session.flush()` triggers INSERT to detect IntegrityError early; commit closes the transaction
- Repository pattern keeps services agnostic of SQLAlchemy specifics
- `dependency_overrides` in FastAPI tests is what makes DI worth the ceremony

**Things I've seen but don't deeply understand yet (and that's OK):**
- SQLAlchemy session lifecycle in detail (when exactly does flush vs commit happen)
- Connection pool tuning (just using defaults)
- Alembic migration conflict resolution (haven't hit it yet)

**Tomorrow:**
- Day 5: relationships between models, more endpoints (GET /users, GET /users/{id})

## Day 5 — [04/16/2026]

**Done:**
- Added updated_at field with onupdate to User model + migration
- Completed CRUD: GET list (paginated), GET by id, PATCH (partial), DELETE
- Built centralized exception handling with PaddingtonError hierarchy
- Single handler catches all domain exceptions automatically
- 12+ tests covering all endpoints, happy and sad paths
- Swagger UI verified end-to-end

**Learned:**
- model_dump(exclude_unset=True) distinguishes "field not sent" from "field sent as null" — critical for PATCH
- Query(ge=1, le=100) validates query params declaratively, same pattern as Field for body
- Centralized exception handlers eliminate try/except duplication in routes
- Base exception with status_code attribute lets one handler manage all domain errors
- onupdate in mapped_column auto-updates timestamps on every UPDATE

**Things I've seen but don't deeply understand yet (and that's OK):**
- SQLAlchemy session internals: flush vs commit timing in nested operations
- How select().order_by().limit().offset() translates to actual SQL
- Alembic migration conflict resolution (still haven't hit it)

**Tomorrow:**
- Day 6: Auth — hashing, JWT, Bearer tokens, login/signup, protected endpoints

## Day 6 — [04/17/2026]

**Done:**
- Studied security theory: encoding, encryption, hashing (bcrypt, scrypt, argon2), and JWT principle of integrity, not confidentiality
- Implemented argon2 password hashing with passlib
- Built POST /auth/signup and POST /auth/login endpoints
- Generated JWTs with python-jose including sub, email, exp, iat claims
- Built get_current_user dependency using HTTPBearer
- Protected GET /users/me with Bearer token
- Moved shared test fixtures to conftest.py
- 23 tests passing

**Key security concepts I can now explain:**
- Why bcrypt and not SHA-256 for passwords (intentionally slow prevents brute force)
- Why JWT payload is readable but secure (signature prevents tampering)
- Why login error messages should be identical for wrong email vs wrong password
- What "Bearer" means in Authorization header (RFC 6750, "whoever bears this token")

**Tomorrow:**
- Day 7: RBAC (roles), refresh tokens, and OAuth conceptual overview

## Day 7 — [04/20/2026]

**Done:**
- Added UserRole enum (user, admin) and role field to User model
- Built require_role dependency factory using closures
- Protected DELETE and list endpoints with admin-only access
- Built PATCH /users/{id}/role for admin promotion
- Created CLI tool for seeding first admin
- Implemented refresh tokens with DB-backed rotation
- POST /auth/refresh revokes old token and issues new pair
- Studied OAuth 2.0 Authorization Code flow conceptually
- Updated all tests for new auth requirements

**Key concepts I can now explain:**
- RBAC: assign permissions to roles, assign roles to users, check roles not individual permissions
- Why require_role returns a function (closure pattern for parameterized dependencies)
- Why refresh tokens exist (short-lived access for security + long-lived refresh for UX)
- Token rotation: revoke on use, detect stolen tokens when legitimate user gets 401
- OAuth 2.0 Authorization Code flow: why the code intermediate step exists (separates browser-visible redirect from server-to-server token exchange)

**Tomorrow:**
- Day 8: week 2 wrap-up, code cleanup, full test suite verification, plan week 3 (LLM APIs + RAG)

## Day 8 — [04/22/2026]

**Done:**
- Applied messages array structure (system, user, assistant, tool roles)
- Called OpenAI and Anthropic using directly its SDK to compare differences
- Implemented streaming for both providers
- Built structured output with Pydantic (OpenAI response_format vs Anthropic tool_use trick)
- Sent images to both LLMs (URL for OpenAI, base64 for Anthropic)
- Built LLMClient wrapper with unified response, retries, cost tracking, structured logging
- Created POST /chat authenticated endpoint
- Wrote tests with AsyncMock (no real API calls)

**Key differences between OpenAI and Anthropic I can now explain:**
- System prompt: in messages array (OpenAI) vs separate parameter (Anthropic)
- Response: string (OpenAI) vs array of content blocks (Anthropic)
- Structured output: native response_format (OpenAI) vs tool_use workaround (Anthropic)
- Image input: URL accepted (OpenAI) vs base64 required (Anthropic)
- max_tokens: optional (OpenAI) vs mandatory (Anthropic)

**Why I'm NOT using LiteLLM:**
- Currently I am using only 2 providers; wrapper is less than 100 lines, not worth a dependency
- Want to demonstrate I understand both APIs directly
- Would use LiteLLM if team had 3+ providers or needed A/B testing

**Tomorrow:**
- Day 9: Embeddings, pgvector setup, chunking, RAG pipeline without frameworks

## Day 9 — [04/23/2026]

**Done:**
- Reflect on RAG theory: what problem it solves, limitations, alternatives (stuffing, GraphRAG, agentic RAG, hybrid search)
- Set up pgvector in Docker Compose with Alembic migration
- Created Document and DocumentChunk models with vector(1536) column
- Built EmbeddingService with batch support (text-embedding-3-small)
- Implemented chunking with RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- Built full RAG pipeline: upload → chunk → embed → store → query → search → LLM → answer
- POST /documents and POST /documents/query endpoints with auth
- Tests with mocked embedding and LLM services
- Verified end-to-end with real Wikipedia article

**Key concepts I can now explain:**
- RAG solves "LLMs don't know your private data" without fine-tuning
- Chunking quality is the #1 factor in RAG quality (more than model choice)
- Embeddings capture semantic similarity, not logical operations (can't search for negations)
- Cosine distance in pgvector is O(n) without index, HNSW index needed for >10k vectors
- Long context windows (200k tokens) are an alternative to RAG for small document sets
- GraphRAG captures entity relationships that vector search misses
- Batch embedding is 100x faster than sequential for multiple texts

**Limitations I'm aware of:**
- No reranking yet (chunks are ranked purely by cosine distance)
- No hybrid search (keyword + semantic)
- Chunking params are not optimized for my specific use case
- No hallucination detection (LLM might still ignore the context)

**Tomorrow:**
- Day 10: More RAG refinements, or begin tool calling if RAG is solid

## Day 10 — [04/24/2026]

**Done:**
- Learned tool calling protocol for OpenAI and Anthropic
- Built manual agent loop from scratch (exercises 11, 12)
- Understood the observe → think → check → act cycle
- Created PaddingtonTools with 3 real tools over pgvector data
- Built production AgentLoop with iteration limits, cost budgets, error recovery
- Created POST /agent/run endpoint with auth protection

**Key concepts I can now explain:**
- The LLM doesn't execute tools — it decides WHICH tool to call with WHAT arguments; your code executes
- tool_call_id links each result to the call that originated it (critical for parallel calls)
- The agent loop is the same pattern for every AI agent: ChatGPT, Claude, Cursor, paddington
- Budget limits prevent runaway costs; iteration limits prevent infinite loops
- Tool errors should be passed to the LLM, not crash the agent

**OpenAI vs Anthropic tool calling differences:**
- Schema format: "parameters" (OpenAI) vs "input_schema" (Anthropic)
- Stop signal: "tool_calls" (OpenAI) vs "tool_use" (Anthropic)
- Result role: "tool" (OpenAI) vs "user" with tool_result blocks (Anthropic)

**Tomorrow:**
- Day 11: LangGraph — refactor the agent loop into a state graph, add persistence

## Day 11 — [04/27/2026]

**Done:**
-  Replaced a hand-rolled LangGraph StateGraph plus raw LiteLLM acompletion calls with the high-level
 langchain.agents.create_agent API backed by ChatLiteLLM.
- Deleted the manual the LangChain ↔ OpenAI dict message converter
- Deleted the custom PaddingtonTools registry
- Tools are now plain @tool-decorated async functions produced by a build_paddington_tools(...) factory that closes over the user's repo and embedding service. 
- The budget gate moved from a dedicated graph node into a BudgetMiddleware(AgentMiddleware) that runs
 after_model.

**Key concepts I can now explain:**
- The old code mixed two paradigms — LangChain message types with raw LiteLLM dicts — which forced a manual
 translation layer in every node and reinvented the prebuilt ReAct loop, ToolNode, and @tool decorator that
 LangGraph and LangChain already ship. 
- More critically, the agent was stateless per-request even though an AsyncPostgresSaver was already created in app.state.checkpointer and never consumed making the agent starting with an empty message history, blocking any multi-turn product feature. 
- The refactor unblocks four capabilities that the framework was always supposed to give:
  - State persistence across HTTP requests (the agent now remembers prior turns)
  - Crash recovery mid-execution via the same checkpoint
  - Future human-in-the-loop interrupts via interrupt() on the same graph
  - Per-user thread isolation without building a permissions table. 
  
- This refactors sets up the next features: multi-turn conversations over uploaded documents, and a future
 Playwright-style multi-step browsing agent — without another rewrite of the loop.

## Day 12 y 13 - [04/30/2026]

**Done:**
- Conversations persist between HTTP requests, with client-supplied thread_ids scoped server-side by user_id for tenant isolation.
- The checkpointer injected via a typed Depends(get_checkpointer) rather than reaching into app.state directly
- Allowed multi-turn conversations enabled end-to-end.
- Bug fixed in the FastAPI lifespan via AsyncExitStack.

**Architecture decisions**
- create_agent (langchain v1) over the deprecated langgraph.prebuilt.create_react_agent to ride the official
 direction and gain the middleware system. 
- Budget enforcement per-turn instead of per-conversation, because cumulative budget breaks predictably on long conversations and the practical question is always "how much did this request cost." 
- runtime.context rather than a state field for the baseline message count, because that value changes
  per invocation and must not be checkpointed — LangGraph distinguishes "per-invocation data" (context) from "graph state" (state) explicitly. 
- Server-side thread scoping via f"{user_id}:{client_thread_id}" instead of an ownership
 table, since it's one line and makes the security invariant impossible to violate without a code change.
- Depends(get_checkpointer) over direct app.state access in routes, to enable app.dependency_overrides for tests, give the route a typed signature, and confine the app.state.checkpointer name to a single source of truth in dependencies.py. 
- AsyncExitStack in the lifespan because AsyncPostgresSaver.from_conn_string is an
 @asynccontextmanager — any other shape either trips Pylance, closes the Postgres connection before requests can use it, or both.