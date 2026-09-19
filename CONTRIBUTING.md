# Contributing to gz-context-engine

Thanks for considering a contribution! This project follows strict quality standards.

## Development Setup

```bash
git clone https://github.com/oke3/gz-context-engine.git
cd gz-context-engine
npm install
npm run dev  # watch mode
```

## Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes with tests
4. Run the full suite: `npm test`
5. Type-check: `npm run lint`
6. Commit with a [conventional commit](https://www.conventionalcommits.org/) message
7. Open a PR against `main`

## Code Standards

- **TypeScript strict mode** — no `any`, no `@ts-ignore`
- **Every public function** gets a JSDoc comment
- **Every module** has a copyright header: `// Copyright (c) 2026 Ground Zero LLC.`
- **Tests required** for new features and bug fixes
- **No new runtime dependencies** without discussion in an issue

## Commit Convention

```
feat: add new embedding provider
fix: handle empty query in search
docs: update API reference
test: add reranker edge cases
chore: update dependencies
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Built by [Ground Zero LLC](https://github.com/oke3)
