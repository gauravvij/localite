# Ctags-Based Code Index for Guidance

## Goal
Replace Strategy B's keyword→filename guessing with a ctags-based symbol→file lookup, so guidance recommends the correct file (e.g., `valuerep.py` for `PersonName3`) instead of keyword-overlap matches (e.g., `dataset.py`).

## Research Summary
- **ctags**: `universal-ctags` is an apt package. `ctags -R -f - --fields=+n` on a repo outputs `tag_name\tfile\tline;"\tkind:language`. Indexes a ~300-file repo in <200ms. Incremental update via `ctags --append -f tags <file>` takes ~2-5ms.
- **Alternatives considered**: `codebase-memory-mcp` (best but requires MCP integration — invasive), `tree-sitter` (accurate but heavier Python dependency), pure keyword scoring (current approach — fails on semantic gaps).
- **Why ctags**: Zero Python deps (system binary), sub-200ms full index, simple tab-separated output to parse, `--append` for incremental updates, handles 50+ languages.

## Approach
Create a `CodeIndex` class wrapping ctags. Integrate into AgentLoop as an optional component. When guidance fires:
1. Extract key identifiers from the **non-lowercased** objective (preserves case for ctags lookup)
2. Query `CodeIndex.lookup(identifier)` for each
3. If found → precise file recommendation
4. If not found → fall through to existing keyword scoring (unchanged)

After `write_file` / `edit_file`, re-index the changed file incrementally.

## Subtasks
1. **Install ctags and verify**: `sudo apt install -y universal-ctags`, run `ctags -R -f - --fields=+n` on a sample dir, confirm output format.
2. **Create `localite/code_index.py`**: `CodeIndex` class with:
   - `__init__(repo_path)` — runs `ctags -R`, parses output into `{tag_name_lower: [(file, line, kind)]}`
   - `lookup(name: str)` — case-insensitive search, returns `[(file, line, kind)]` sorted: Python files first, class definitions before function/variable
   - `reindex()` — full re-run of ctags
   - `reindex_file(filepath)` — incremental re-index for a changed file
   - `__repr__` for debug logging
   - Verify with `py_compile` and a quick Python import + test
3. **Modify `AgentLoop.__init__`**: Add optional `code_index: Optional['CodeIndex'] = None` parameter. Store as `self.code_index`.
4. **Modify Strategy B (lines ~703-770)** in `agent_loop.py`: Before existing keyword scoring, try ctags lookup:
   - Extract terms from **non-lowercased** objective
   - For each term, call `self.code_index.lookup(term)`
   - If match found, format `[GUIDANCE] 'PersonName3' is defined in valuerep.py. Read it with read_file.` and set `guidance_msg`
   - If no match, fall through to existing keyword score logic
   - Wrap in `if self.code_index is not None:` guard
5. **Modify tool dispatch (line ~1162-1168)**: After successful `write_file`/`edit_file`, call `self.code_index.reindex_file(file_path)` if `self.code_index` is set.
6. **Modify `create_swe_agent()` in `swe_runner.py`**: After setting all tool workdirs, create `CodeIndex(workdir)`, pass to `AgentLoop(code_index=...)`.
7. **Test the integration**: Run a dry eval of pydicom-1139 (or just verify guidance text is correct by asserting the ctags lookup returns `valuerep.py` for `PersonName3`).

## Deliverables
| File | Description |
|------|-------------|
| `localite/code_index.py` | New CodeIndex class wrapping ctags |
| `localite/loop/agent_loop.py` | Modified: optional code_index param, ctags-first guidance, post-edit re-index |
| `swe_runner.py` | Modified: creates CodeIndex and passes to AgentLoop |

## Evaluation Criteria
- `CodeIndex.lookup("PersonName3")` returns `valuerep.py` as the top result
- Strategy B guidance for pydicom-1139 recommends `valuerep.py`, not `dataset.py`
- Existing behavior unchanged when:
  - ctags is not installed (`CodeIndex.__init__` raises/catches gracefully)
  - `code_index` parameter is not passed (None → no-op)
  - No key term matches ctags index (falls through to keyword scoring)
- Post-write/edit_file re-index updates the index without full re-scan

## Notes
- ctags must be installed before running swe_runner — handle `FileNotFoundError` gracefully in CodeIndex (disable index, log warning)
- The incremental `--append` approach works but requires an existing tags file; for simplicity, run full `ctags -R` on every reindex (still <200ms for small repos)