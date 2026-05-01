import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "dist");
const port = Number(process.env.PORT || 4173);

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
};

async function send(res, file) {
  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}

createServer(async (req, res) => {
  const cleanUrl = decodeURIComponent((req.url || "/").split("?")[0]);
  const rel = cleanUrl === "/" ? "/index.html" : cleanUrl;
  const target = path.resolve(root, `.${rel}`);

  if (!target.startsWith(root)) {
    res.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }

  try {
    const info = await stat(target);
    if (info.isFile()) {
      await send(res, target);
      return;
    }
  } catch {
    // Fall through to SPA entry.
  }

  await send(res, path.join(root, "index.html"));
}).listen(port, "0.0.0.0", () => {
  console.log(`ConcordAI frontend: http://localhost:${port}`);
});
