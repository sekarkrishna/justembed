# Changelog

All notable changes to JustEmbed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1a7] - 2026-02-19

### Added
- Background job system with SQLite persistence for long-running tasks
- Job management UI with progress tracking and status monitoring
- Progress callbacks for model training with 5-stage reporting
- Job status page with auto-refresh for real-time updates
- Background worker for async job processing
- Upload history tracking in database
- CODE_OF_CONDUCT.md following Contributor Covenant v2.1
- CI/CD workflows for GitHub Actions
  - Continuous Integration with multi-version Python testing
  - Automated PyPI publishing on release
  - Security scanning with bandit and safety
  - Code quality checks with ruff, black, and mypy

### Changed
- `/train-model` endpoint now uses background jobs instead of blocking
- `/embed` endpoint now uses background jobs for chunk embedding
- Model training shows progress in real-time via job status page
- Improved error handling for job failures
- Updated documentation footer with docs link

### Fixed
- Version number consistency across package files
- Port number documentation (confirmed 5424 as standard)

## [0.1.1a6] - 2026-02-16

### Added
- Documentation site live at https://sekarkrishna.github.io/justembed
- Comprehensive API documentation
- Usage examples and tutorials

### Changed
- Improved README with better examples
- Enhanced error messages

## [0.1.1a5] - 2026-02-15

### Added
- Custom model training with TF-IDF→MLP pipeline
- ONNX export for custom models
- Model persistence and loading
- Custom embedder support

### Changed
- Refactored embedder architecture for pluggability
- Improved chunking algorithm

### Fixed
- Memory leaks in long-running server
- Edge cases in text chunking

## [0.1.1a4] - 2026-02-14

### Added
- Multiple knowledge base support
- KB-specific model selection
- Upload history tracking

### Changed
- Database schema improvements
- Better error handling

## [0.1.1a3] - 2026-02-13

### Added
- Web UI for chunk preview
- Configurable chunking parameters
- File upload validation

### Fixed
- Chunking edge cases
- UI responsiveness issues

## [0.1.1a2] - 2026-02-12

### Added
- Python API for programmatic access
- Workspace management
- Knowledge base CRUD operations

### Changed
- Improved CLI interface
- Better documentation

## [0.1.1a1] - 2026-02-11

### Added
- Initial alpha release
- E5-Small ONNX embedder
- DuckDB vector storage
- FastAPI web interface
- Basic semantic search functionality
- Offline-first architecture

---

## Release Notes

### Alpha Release Status

JustEmbed is currently in **alpha** status. This means:

- ✅ Core functionality is working and tested
- ✅ API is functional but may change
- ⚠️ Not recommended for production use yet
- ⚠️ Test coverage is limited
- ⚠️ Breaking changes may occur between versions

### Roadmap to v1.0 (August 2026)

Planned features for stable release:

- Comprehensive test suite (>80% coverage)
- Stable API (no breaking changes)
- Research vs. Production modes
- Model registry with 10+ curated models
- Enterprise license file support
- Docker deployment support
- Polars integration for performance
- JOSS publication

### Feedback Welcome

This is an early alpha release. We're actively seeking feedback on:

- Bug reports
- Feature requests
- Use case ideas
- Documentation improvements
- Performance issues

Please report issues at: https://github.com/sekarkrishna/justembed/issues

---

## Version Numbering

JustEmbed follows semantic versioning with alpha releases:

- `0.1.1a7` = Version 0.1.1, alpha release 7
- `0.1.2a1` = Version 0.1.2, alpha release 1
- `0.1.2` = Version 0.1.2, stable release

Alpha releases may have breaking changes. Stable releases will maintain backward compatibility.

---

**Maintained by**: Krishnamoorthy Sankaran  
**Contact**: krishnamoorthy.sankaran@sekrad.org  
**Repository**: https://github.com/sekarkrishna/justembed
