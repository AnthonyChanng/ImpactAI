import { test, expect } from "@playwright/test";
const live = process.env.E2E_DATASET === "1";
const playerName = live ? "Reduxx" : "Example Player";

test("unavailable round estimates stay unavailable in the chart", async ({page}) => {
  await page.route("**/api/matches/60894/rounds", async route => {
    const response = await route.fetch();
    const body = await response.json();
    body.data = body.data.map((round: {probability: number | null}) => ({...round, probability: null}));
    await route.fulfill({response, json: body});
  });
  await page.goto("/matches/60894");
  await expect(page.getByText("Unavailable", {exact: true})).toBeVisible();
  const chart = page.getByRole("img", {name: /^Team 1 pre-round probability/});
  await expect(chart.locator("svg.recharts-surface")).toBeVisible();
  await expect(chart.locator(".recharts-area-curve")).toHaveCount(0);
});

test("changing match cancels a pending coaching report", async ({page}) => {
  let release: () => void = () => {};
  const delayed = new Promise<void>(resolve => {release = resolve;});
  let captured: () => void = () => {};
  const capturedResponse = new Promise<void>(resolve => {captured = resolve;});
  await page.route("**/api/analysis/coach", async route => {
    const response = await route.fetch();
    captured();
    await delayed;
    await route.fulfill({response});
  }, {times: 1});
  await page.goto("/coach?player=8419&game=60894");
  const match = page.getByRole("combobox", {name: "Select a match"});
  await expect(match).toBeEnabled();
  await page.getByRole("button", {name: "Analyze performance"}).click();
  await capturedResponse;
  const otherGame = await match.locator("option").evaluateAll(options =>
    (options as HTMLOptionElement[]).find(option => option.value !== "60894" && option.value)?.value
  );
  expect(otherGame).toBeTruthy();
  await match.selectOption(otherGame!);
  release();
  await expect(page.getByRole("button", {name: "Analyze performance"})).toBeEnabled();
  await expect(page.getByText("PERFORMANCE REVIEW", {exact: true})).toHaveCount(0);
  await page.getByRole("button", {name: "Analyze performance"}).click();
  await expect(page.getByText(`Evidence: game #${otherGame}`, {exact: false})).toHaveCount(3);
  await expect(page.getByText("Evidence: game #60894", {exact: false})).toHaveCount(0);
  await expect(page.getByRole("link", {name: "Source match"})).toHaveAttribute("href", `/matches/${otherGame}?player=8419`);
});

test("search historical player, select game, inspect analysis and evidence", async ({page}) => {
  await page.goto("/players");
  await page.getByRole("textbox", {name: "Search players"}).fill(playerName);
  await page.getByRole("link", {name: `View ${playerName}`, exact: true}).click();
  await expect(page.getByRole("heading", {name: playerName, exact: true})).toBeVisible();
  await page.getByRole("button", {name: "Analyze game 60894", exact: true}).click();
  await expect(page.getByText("Inspect a performance")).toBeVisible();
  await page.getByRole("link", {name: "Full match"}).click();
  await expect(page.getByRole("heading", {name: "Breeze / Match analysis"})).toBeVisible();
  await page.getByRole("button", {name: "Round 2:", exact: false}).click();
  await expect(page.getByText("ROUND 2", {exact: true})).toBeVisible();
  await expect(page.getByText("Unknown (overtime)")).toHaveCount(0);
  await page.getByRole("link", {name: "Coach", exact: true}).click();
  await page.getByRole("button", {name: "Analyze performance"}).click();
  await expect(page.getByText("PERFORMANCE REVIEW", {exact: true})).toBeVisible();
  await expect(page.getByText("Evidence: game #60894", {exact: false})).toHaveCount(3);
});

test("overview and transparent metrics show expected counts and no horizontal page overflow", async ({page}) => {
  await page.goto("/");
  await expect(page.locator(".stat-card").filter({hasText: "Games analyzed"}).locator("strong")).toHaveText(live ? "13,228" : "2");
  await expect(page.locator(".stat-card").filter({hasText: "Players in the dataset"}).locator("strong")).toHaveText(live ? "7,211" : "10");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  expect(overflow).toBe(false);
  await page.goto("/models");
  await expect(page.getByRole("heading", {name: "Predeclared model evaluation"})).toBeVisible();
  await page.getByRole("tab", {name: "Postgame player association"}).click();
  await expect(page.getByText("diagnostic controls only", {exact: true})).toBeVisible();
  await page.goto("/agents");
  await page.getByRole("combobox", {name: "Filter agents by map"}).selectOption("Ascent");
  await expect(page.getByRole("heading", {name: "Historical agent comparison"})).toBeVisible();
});
