"use client";
import { useEffect, useState } from "react";
import type { Envelope } from "@/types/api";

const cache = new Map<string, {response: Envelope<unknown>; expires: number}>();
const pending = new Map<string, Promise<Envelope<unknown>>>();
const MAX_ENTRIES = 100;

async function request(url: string): Promise<Envelope<unknown>> {
  const cached = cache.get(url);
  if (cached && cached.expires > Date.now()) return cached.response;
  const existing = pending.get(url);
  if (existing) return existing;
  const promise = (async () => {
    const response = await fetch(url, {signal: AbortSignal.timeout(15000)});
    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error(body?.error?.message || "Unable to load data. Please retry.");
    if (!body || typeof body !== "object" || !("data" in body) || !body.meta) throw new Error("Unexpected data response. Please retry.");
    cache.delete(url);
    cache.set(url, {response: body, expires: Date.now() + 120000});
    if (cache.size > MAX_ENTRIES) cache.delete(cache.keys().next().value!);
    return body;
  })();
  pending.set(url, promise);
  try { return await promise; }
  finally { pending.delete(url); }
}

export function useApi<T>(url: string | null, {keepPreviousData = false} = {}) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<{key: string; response?: Envelope<T>; error?: string}>({key: ""});
  const key = `${url}:${attempt}`;
  useEffect(() => {
    if (!url) return;
    let active = true;
    // Shared requests finish for other subscribers; obsolete subscribers cannot update their views.
    request(url).then(response => {
      if (active) setState({key, response: response as Envelope<T>});
    }).catch(error => {
      if (active) setState({key, error: error.name === "TimeoutError" ? "Request timed out. Please retry." : error.message});
    });
    return () => { active = false; };
  }, [url, key, attempt]);
  const current = state.key === key;
  const response = url && (current || keepPreviousData) ? state.response : undefined;
  const loading = !!url && (!current || (!state.response && !state.error));
  return {data: response?.data, meta: response?.meta,
    error: current ? state.error : undefined, loading,
    refreshing: loading && !!response, retry: () => { if (url) cache.delete(url); setAttempt(a => a + 1); }};
}
