import { build } from "esbuild";
import { mkdir, copyFile } from "node:fs/promises";
await mkdir("dist", { recursive: true });
await build({
  entryPoints: ["app.js"],
  bundle: true,
  format: "esm",
  minify: true,
  outfile: "dist/app.js",
});
for (const file of ["index.html", "style.css"])
  await copyFile(file, `dist/${file}`);
