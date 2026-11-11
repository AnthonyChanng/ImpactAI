export const number = (value: number, digits = 0) => value.toLocaleString("en-US", {maximumFractionDigits: digits, minimumFractionDigits: digits});
export const percent = (value: number, digits = 1) => `${number(value * 100, digits)}%`;
export const date = (value: string) => new Date(`${value.slice(0, 10)}T12:00:00Z`).toLocaleDateString("en-US", {month: "short", day: "numeric", year: "numeric", timeZone: "UTC"});
export const featureName = (value: string) => value.replace(/^(numeric|category)__/, "").replaceAll("_", " ");
export const initials = (value: string) => value.slice(0, 2).toUpperCase();
