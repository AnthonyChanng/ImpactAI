import { cpSync } from "node:fs";
import { fileURLToPath } from "node:url";

const base = new URL("../", import.meta.url);
cpSync(new URL(".next/static", base), new URL(".next/standalone/.next/static", base), {recursive: true});
process.env.HOSTNAME = "127.0.0.1";
process.env.NODE_ENV = "production";
process.chdir(fileURLToPath(base));
await import(new URL(".next/standalone/server.js", base).href);
