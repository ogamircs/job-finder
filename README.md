# Job Finder

Local Gradio app that ingests a resume PDF or Reactive Resume profile, searches Google Jobs through SerpApi, and ranks the best matches with OpenAI.

## Requirements

- Python 3.11
- `uv`
- An OpenAI API key
- A SerpApi key
- Optional: a Reactive Resume base URL and API key

## Run

```bash
uv sync --python 3.11 --all-groups
uv run job-finder
```

## Flow

1. On the first launch, the app opens a setup screen.
2. Enter your API keys once. They are written to `.env` in the project root.
3. Either upload a PDF to store in `.resume/` or save your Reactive Resume API details.
4. After setup, the app switches to the job search workspace.
5. On later launches, if `.env` already exists, the app skips setup and opens the search workspace directly.
6. Use the `Settings` button in the top-right of the search workspace to update API keys or the Reactive Resume URL and save them back to `.env`.
7. Upload a PDF by drag-and-drop or with the `Browse PDF` button.
8. Click `Analyze resume` to prefill the inferred location and editable search terms.
9. Adjust the location and search terms if needed, then click `Find jobs`.

## Storage

- `.env` stores `OPENAI_API_KEY`, `SERPAPI_API_KEY`, `RX_RESUME_API_KEY`, and `RX_RESUME_API_URL`.
- `.resume/` stores uploaded PDF resumes so they can be reused from the dropdown.
- The default Reactive Resume endpoint is `https://rxresu.me/api/openapi/resumes`.

## Notes

- The search workspace reads secrets from `.env` automatically. You do not need to `source` the file before launching the app.
- Search terms are generated from the analyzed resume, but you can edit them before running the job search.
- `Find jobs` stays disabled until the currently selected resume has been analyzed.
- The results table includes a `pay_range` column when salary data is present in SerpApi results.

## Support

If you enjoy this, buy me some tokens: [buymeacoffee.com/amircs](https://buymeacoffee.com/amircs)
