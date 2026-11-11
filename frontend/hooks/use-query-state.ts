"use client";
import { useSearchParams } from "next/navigation";

// Read the URL on every render; native history updates also support Back/Forward.
export function useQueryState() {
  const params = useSearchParams();
  function update(values: Record<string, string | number>, replace = false) {
    const url = new URL(window.location.href);
    for (const [key, value] of Object.entries(values)) {
      if (value === "" || (key === "page" && value === 1)) url.searchParams.delete(key);
      else url.searchParams.set(key, String(value));
    }
    const next = `${url.pathname}${url.search}${url.hash}`;
    if (next !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.history[replace ? "replaceState" : "pushState"](null, "", next);
    }
  }
  const rawPage = Number(params.get("page") || 1);
  const page = Number.isSafeInteger(rawPage) && rawPage > 0 ? rawPage : 1;
  return {params, update, page};
}
