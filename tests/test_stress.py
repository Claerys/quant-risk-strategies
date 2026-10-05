import pandas as pd
import pytest

from quant_risk.stress import (
    Scenario,
    load_scenarios,
    reverse_stress,
    run_scenario,
    sector_shock_grid,
    stress_table,
)

DATES = pd.bdate_range("2024-01-01", periods=10)
CLOSES = pd.DataFrame(
    {"ES": [5000.0 + 10 * i for i in range(10)], "C": [400.0] * 5 + [450.0] * 5, "CL": [70.0] * 10},
    index=DATES,
)


def test_shipped_scenarios_load_and_cover_three_categories() -> None:
    scenarios = load_scenarios()
    assert {s.category for s in scenarios} == {"historical", "hypothetical", "climate"}
    assert all(s.description for s in scenarios)


def test_unknown_sector_in_a_scenario_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('[[scenario]]\nname="x"\nkind="shock"\n[scenario.shocks]\nmoon = 5\n')
    with pytest.raises(ValueError, match="moon"):
        load_scenarios(path)


def test_symbol_shock_overrides_its_sector() -> None:
    scenario = Scenario("t", "climate", "shock", "", shocks={"energy": -20.0, "NG": 10.0})
    assert scenario.shock_for("CL") == -20.0
    assert scenario.shock_for("NG") == 10.0
    assert scenario.shock_for("ES") == 0.0


def test_percentage_shock_is_applied_to_todays_price() -> None:
    scenario = Scenario("crash", "hypothetical", "shock", "", shocks={"us_equity_index": -10.0})
    # 2 ES at 5090: -10% = -509 points x $50 x 2 = -50,900.
    result = run_scenario(scenario, pd.Series({"ES": 2.0, "C": 0.0, "CL": 0.0}), CLOSES)
    assert result.pnl == pytest.approx(-50_900.0)


def test_historical_replay_uses_real_price_changes() -> None:
    scenario = Scenario("drought", "climate", "historical", "", start=DATES[2], end=DATES[7])
    # Short 3 corn while corn rises 50 cents: -50 x $50 x 3 = -7,500. ES +50 points on 1 long = +2,500.
    result = run_scenario(scenario, pd.Series({"ES": 1.0, "C": -3.0, "CL": 0.0}), CLOSES)
    assert result.by_instrument["C"] == pytest.approx(-7_500.0)
    assert result.pnl == pytest.approx(-5_000.0)


def test_stress_table_reports_share_of_capital() -> None:
    scenario = Scenario("crash", "hypothetical", "shock", "", shocks={"us_equity_index": -10.0})
    table = stress_table(pd.Series({"ES": 2.0, "C": 0.0, "CL": 0.0}), CLOSES, capital=1_000_000, scenarios=[scenario])
    assert table.loc["crash", "% of capital"] == pytest.approx(-0.0509)
    assert "ES" in table.loc["crash", "biggest losses"]


def test_reverse_stress_finds_the_worst_window() -> None:
    pnl = pd.DataFrame({"ES": [1.0, -5.0, -6.0, 2.0, 1.0]}, index=DATES[:5])
    worst = reverse_stress(pd.Series({"ES": 1.0}), pnl, horizons=(1, 2))
    assert worst.loc[1, "worst P&L"] == -6.0
    assert worst.loc[2, "worst P&L"] == -11.0


def test_sector_grid_is_linear_in_the_move() -> None:
    grid = sector_shock_grid(pd.Series({"ES": 1.0, "C": 0.0, "CL": -2.0}), CLOSES)
    assert grid.loc["energy", "+10%"] == pytest.approx(-2 * 70 * 1000 * 0.10)
    assert grid.loc["us_equity_index", "-20%"] == pytest.approx(-2 * grid.loc["us_equity_index", "+10%"])
    assert "grains" not in grid.index  # no position, no row
