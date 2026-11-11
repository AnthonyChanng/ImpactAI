import { test, expect } from "@playwright/test";

test("search waits for typing, retains results, and reuses a cached filter", async ({page}) => {
  const requests: string[] = [];
  page.on("request", request => {
    const url = new URL(request.url());
    if (url.pathname === "/api/players") requests.push(url.searchParams.get("q") || "");
  });
  await page.clock.install({time: new Date("2026-01-01T00:00:00Z")});
  await page.goto("/players");
  await expect(page.getByRole("link", {name: "View Example Player", exact: true})).toBeVisible();
  let release: () => void = () => {};
  const gate = new Promise<void>(resolve => {release = resolve;});
  await page.route("**/api/players?**", async route => {
    if (new URL(route.request().url()).searchParams.get("q") === "Example Player") {
      await gate;
    }
    await route.continue();
  });
  await page.clock.pauseAt(new Date("2026-01-01T00:00:30Z"));
  const search = page.getByRole("textbox", {name: "Search players"});
  await search.fill("Ex");
  await page.clock.runFor(100);
  await search.fill("Example Player");
  await expect(page).toHaveURL(/q=Example\+Player/);
  await page.clock.runFor(300);
  await expect.poll(() => requests.includes("Example Player")).toBe(true);
  await expect(page.getByRole("status")).toContainText("Updating results");
  await expect(page.getByRole("link", {name: "View Example Player", exact: true})).toBeVisible();
  expect(requests).not.toContain("Ex");
  await page.clock.resume();
  release();
  await expect(page.getByRole("status")).toBeEmpty();
  await search.fill("");
  await page.clock.runFor(300);
  await expect(page.getByRole("status")).toBeEmpty();
  expect(requests.filter(q => q === "")).toHaveLength(1);
  expect(requests.filter(q => q === "Example Player")).toHaveLength(1);
});

test("filters survive reload and browser back after a selection", async ({page}) => {
  await page.goto("/players?agent=jett&map=Breeze");
  const map = page.getByRole("combobox", {name: "Filter by map"});
  await map.selectOption("Ascent");
  await expect(page).toHaveURL(/map=Ascent/);
  await page.reload();
  await expect(map).toHaveValue("Ascent");
  await page.goBack();
  await expect(map).toHaveValue("Breeze");
  await page.goto("/matches?map=Ascent&event=Browser&date_from=2021-01-01");
  await expect(page.getByRole("combobox", {name: "Filter by map"})).toHaveValue("Ascent");
  await expect(page.getByRole("textbox", {name: "Search events"})).toHaveValue("Browser");
  await expect(page.getByLabel("From date", {exact: true})).toHaveValue("2021-01-01");
});

test("peer ranges show measured coverage and explain missing research observations", async ({page}, testInfo) => {
  await page.goto("/players/8419?game=60894");
  const card = page.getByRole("heading", {name: "Observed performance and peer references"}).locator("xpath=ancestor::section");
  await expect(card.getByRole("columnheader", {name: "Measured coverage"})).toBeVisible();
  await expect(card.getByText("78.0%", {exact: true})).toHaveCount(6);
  await page.screenshot({path: testInfo.outputPath("peer-ranges.png"), fullPage: true});
  await page.getByRole("button", {name: "Analyze game 60895"}).click();
  await expect(page).toHaveURL(/game=60895/);
  await expect(card.getByText(/outside the saved research sample/)).toBeVisible();
  await expect(card.getByRole("columnheader", {name: "Measured coverage"})).toHaveCount(0);
});

test("non-JSON API failures give a useful retry and recover", async ({page}) => {
  await page.route("**/api/players?**", route => route.fulfill({status: 502, contentType: "text/html", body: "Bad Gateway"}), {times: 1});
  await page.goto("/players");
  await expect(page.getByRole("alert").filter({hasText: "Data is unavailable"})).toContainText("Unable to load data");
  await page.getByRole("button", {name: "Retry", exact: true}).click();
  await expect(page.getByRole("link", {name: "View Example Player", exact: true})).toBeVisible();
});

test.beforeEach(() => {
  test.skip(process.env.E2E_DATASET === "1", "These assertions target artificial fixtures only.");
});
