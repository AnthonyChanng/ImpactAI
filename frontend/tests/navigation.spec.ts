import { test, expect } from "@playwright/test";

test("player filters follow query navigation and browser back", async ({page}) => {
  await page.goto("/players?agent=jett");
  const agent = page.getByRole("combobox", {name: "Filter by agent"});
  await expect(agent).toHaveValue("jett");
  await page.evaluate(() => window.history.pushState(null, "", "/players?agent=sage&map=Ascent"));
  await expect(agent).toHaveValue("sage");
  await expect(page.getByRole("combobox", {name: "Filter by map"})).toHaveValue("Ascent");
  await page.goBack();
  await expect(agent).toHaveValue("jett");
  await expect(page.getByRole("combobox", {name: "Filter by map"})).toHaveValue("");
});

test("match breakdown follows player links after query navigation", async ({page}) => {
  await page.goto("/matches/60894?player=8419");
  const player = page.getByRole("combobox", {name: "Select player for breakdown"});
  await expect(player).toHaveValue("8419");
  const otherPlayer = await player.locator("option").evaluateAll(options =>
    (options as HTMLOptionElement[]).find(option => option.value !== "8419")?.value
  );
  expect(otherPlayer).toBeTruthy();
  await page.evaluate(id => window.history.pushState(null, "", `/matches/60894?player=${id}`), otherPlayer);
  await expect(player).toHaveValue(otherPlayer!);
  await expect(page.getByRole("link", {name: "Coach", exact: true})).toHaveAttribute("href", `/coach?player=${otherPlayer}&game=60894`);
  await page.goBack();
  await expect(player).toHaveValue("8419");
});

test("coach clears old evidence when the linked performance changes", async ({page}) => {
  await page.goto("/coach?player=8419&game=60894");
  const match = page.getByRole("combobox", {name: "Select a match"});
  await expect(match).toBeEnabled();
  await page.getByRole("button", {name: "Analyze performance"}).click();
  await expect(page.getByText("PERFORMANCE REVIEW", {exact: true})).toBeVisible();
  const otherGame = await match.locator("option").evaluateAll(options =>
    (options as HTMLOptionElement[]).find(option => option.value !== "60894" && option.value)?.value
  );
  expect(otherGame).toBeTruthy();
  await page.evaluate(id => window.history.pushState(null, "", `/coach?player=8419&game=${id}`), otherGame);
  await expect(match).toHaveValue(otherGame!);
  await expect(page.getByText("PERFORMANCE REVIEW", {exact: true})).toHaveCount(0);
  await page.getByRole("button", {name: "Analyze performance"}).click();
  await expect(page.getByRole("link", {name: "Source match"})).toHaveAttribute("href", `/matches/${otherGame}?player=8419`);
});

test("agent drilldown retains its selected map", async ({page}) => {
  await page.goto("/agents");
  await page.getByRole("combobox", {name: "Filter agents by map"}).selectOption("Ascent");
  const drilldown = page.getByRole("link", {name: "Compare jett players"});
  await expect(drilldown).toHaveAttribute("href", "/players?agent=jett&map=Ascent");
  await drilldown.click();
  await expect(page.getByRole("combobox", {name: "Filter by agent"})).toHaveValue("jett");
  await expect(page.getByRole("combobox", {name: "Filter by map"})).toHaveValue("Ascent");
});

test("agent distribution falls back when the selected agent has no map records", async ({page}) => {
  await page.route("**/api/agents?map=Ascent", async route => {
    const response = await route.fetch();
    const body = await response.json();
    body.data = body.data.filter((agent: {name: string}) => agent.name === "sage");
    await route.fulfill({response, json: body});
  });
  await page.goto("/agents");
  await expect(page.getByRole("combobox", {name: "Distribution agent"})).toHaveValue("jett");
  await page.getByRole("combobox", {name: "Filter agents by map"}).selectOption("Ascent");
  const distribution = page.getByRole("heading", {name: "Contextual performance distribution"}).locator("xpath=ancestor::section");
  await expect(distribution.getByRole("combobox", {name: "Distribution agent"})).toHaveValue("sage");
  await expect(distribution.locator(".recharts-rectangle").first()).toBeVisible();
});
