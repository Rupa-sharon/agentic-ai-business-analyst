import sys, markdown

src = sys.argv[1] if len(sys.argv) > 1 else "reports/analysis_1.md"
out = src.replace(".md", ".html")

with open(src, encoding="utf-8") as f:
    body = markdown.markdown(f.read(), extensions=["tables"])

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Business report</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; padding: 0 16px; }}
h1, h2 {{ color: #1f3a5f; }}
</style></head><body>{body}</body></html>"""

with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("Saved", out)